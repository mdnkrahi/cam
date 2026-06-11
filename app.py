from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import cv2
import logging
from datetime import datetime, date
from io import BytesIO
import os
import numpy as np

# Import configurations and modules
from config import get_config
from models import db, init_db
from models.student import Student, AttendanceRecord
from modules.yolo_detector import YOLODetector
from modules.face_recognizer import FaceRecognizer
from modules.attendance_manager import AttendanceManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load configuration
config = get_config()
app.config.from_object(config)

# Initialize database
init_db(app)

# Initialize modules
detector = YOLODetector()
recognizer = FaceRecognizer()
attendance_manager = AttendanceManager()

# Create necessary directories
os.makedirs('data/student_faces', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# ==================== Routes ====================

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/register')
def register():
    """Student registration page"""
    return render_template('register.html')

@app.route('/api/register-student', methods=['POST'])
def api_register_student():
    """Register a new student"""
    try:
        data = request.form
        file = request.files.get('face_image')
        
        # Validate input
        required_fields = ['student_id', 'name', 'email', 'class_name']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'Missing field: {field}'}), 400
        
        # Check if student already exists
        existing = Student.query.filter_by(student_id=data['student_id']).first()
        if existing:
            return jsonify({'error': 'Student ID already exists'}), 409
        
        # Create student record
        student = Student(
            student_id=data['student_id'],
            name=data['name'],
            email=data['email'],
            phone=data.get('phone', ''),
            class_name=data['class_name']
        )
        
        db.session.add(student)
        db.session.flush()  # Get the student ID
        
        # Process face image if provided
        if file:
            # Read and encode image
            img_array = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
            recognizer.register_student_face(student.id, img_array)
        
        db.session.commit()
        recognizer.refresh_known_faces()
        
        return jsonify({
            'message': 'Student registered successfully',
            'student_id': student.id
        }), 201
    
    except Exception as e:
        logger.error(f"Error registering student: {str(e)}")
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@app.route('/dashboard')
def dashboard():
    """Dashboard page"""
    return render_template('dashboard.html')

@app.route('/api/attendance')
def get_attendance():
    """Get attendance records"""
    try:
        date_str = request.args.get('date', str(date.today()))
        class_name = request.args.get('class_name')
        
        target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        records = attendance_manager.get_attendance_by_date(target_date, class_name)
        
        return jsonify([record.to_dict() for record in records])
    
    except Exception as e:
        logger.error(f"Error fetching attendance: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/student/<int:student_id>/attendance')
def get_student_attendance(student_id):
    """Get attendance history for a student"""
    try:
        days = request.args.get('days', 30, type=int)
        from datetime import timedelta
        
        start_date = date.today() - timedelta(days=days)
        end_date = date.today()
        
        records = attendance_manager.get_attendance_by_student(
            student_id, start_date, end_date
        )
        
        return jsonify([record.to_dict() for record in records])
    
    except Exception as e:
        logger.error(f"Error fetching student attendance: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/student/<int:student_id>/stats')
def get_student_stats(student_id):
    """Get attendance statistics for a student"""
    try:
        stats = attendance_manager.get_attendance_stats(student_id)
        return jsonify(stats)
    
    except Exception as e:
        logger.error(f"Error fetching student stats: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/class/<class_name>/stats')
def get_class_stats(class_name):
    """Get attendance statistics for a class"""
    try:
        stats = attendance_manager.get_class_attendance_stats(class_name)
        return jsonify(stats)
    
    except Exception as e:
        logger.error(f"Error fetching class stats: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/detect-and-mark', methods=['POST'])
def detect_and_mark():
    """Detect faces and mark attendance"""
    try:
        file = request.files.get('image')
        if not file:
            return jsonify({'error': 'No image provided'}), 400
        
        # Read image
        img_array = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        # Detect faces
        detections = detector.detect_faces(frame)
        detections = detector.filter_detections(detections)
        
        if not detections:
            return jsonify({'message': 'No faces detected'}), 200
        
        results = []
        
        # Process each detection
        for detection in detections:
            face_roi = detector.get_face_roi(frame, detection)
            
            # Recognize face
            recognized = recognizer.recognize_faces(face_roi)
            
            if recognized:
                best_match = recognized[0]
                
                # Mark attendance
                attendance_manager.mark_attendance(
                    best_match['student_id'],
                    best_match['confidence']
                )
                
                results.append({
                    'name': best_match['name'],
                    'student_id': best_match['student_id'],
                    'confidence': best_match['confidence'],
                    'marked': True
                })
            else:
                results.append({
                    'name': 'Unknown',
                    'confidence': detection['confidence'],
                    'marked': False
                })
        
        return jsonify({'detections': results}), 200
    
    except Exception as e:
        logger.error(f"Error in detect_and_mark: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/students')
def get_students():
    """Get all students"""
    try:
        class_name = request.args.get('class_name')
        
        query = Student.query.filter_by(is_active=True)
        if class_name:
            query = query.filter_by(class_name=class_name)
        
        students = query.all()
        
        return jsonify([student.to_dict() for student in students])
    
    except Exception as e:
        logger.error(f"Error fetching students: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/export-attendance', methods=['GET'])
def export_attendance():
    """Export attendance to CSV"""
    try:
        class_name = request.args.get('class_name')
        date_str = request.args.get('date')
        
        if date_str:
            target_date = datetime.strptime(date_str, '%Y-%m-%d').date()
            start_date = target_date
            end_date = target_date
        else:
            start_date = None
            end_date = None
        
        csv_data = attendance_manager.export_attendance_to_csv(
            class_name, start_date, end_date
        )
        
        if not csv_data:
            return jsonify({'error': 'No data to export'}), 400
        
        # Create BytesIO object
        output = BytesIO()
        output.write(csv_data.encode('utf-8'))
        output.seek(0)
        
        return send_file(
            output,
            mimetype='text/csv',
            as_attachment=True,
            download_name=f'attendance_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        )
    
    except Exception as e:
        logger.error(f"Error exporting attendance: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ==================== Error Handlers ====================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ==================== Initialization ====================

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=app.config['DEBUG']
    )