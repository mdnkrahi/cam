from datetime import datetime
from models import db

class Student(db.Model):
    __tablename__ = 'students'
    
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.String(50), unique=True, nullable=False, index=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=True)
    phone = db.Column(db.String(20), nullable=True)
    class_name = db.Column(db.String(50), nullable=False)
    face_encoding = db.Column(db.LargeBinary, nullable=True)  # Stored face embedding
    enrollment_date = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationships
    attendance_records = db.relationship('AttendanceRecord', backref='student', lazy=True, cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Student {self.name}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'student_id': self.student_id,
            'name': self.name,
            'email': self.email,
            'phone': self.phone,
            'class_name': self.class_name,
            'enrollment_date': self.enrollment_date.isoformat(),
            'is_active': self.is_active
        }

class AttendanceRecord(db.Model):
    __tablename__ = 'attendance_records'
    
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('students.id'), nullable=False, index=True)
    date = db.Column(db.Date, nullable=False, index=True)
    check_in_time = db.Column(db.DateTime, nullable=False)
    check_out_time = db.Column(db.DateTime, nullable=True)
    confidence_score = db.Column(db.Float, default=0.0)
    status = db.Column(db.String(20), default='present')  # present, absent, late
    notes = db.Column(db.String(255), nullable=True)
    
    def __repr__(self):
        return f'<AttendanceRecord {self.student_id} - {self.date}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'student_id': self.student_id,
            'student_name': self.student.name,
            'date': self.date.isoformat(),
            'check_in_time': self.check_in_time.isoformat(),
            'check_out_time': self.check_out_time.isoformat() if self.check_out_time else None,
            'confidence_score': round(self.confidence_score, 4),
            'status': self.status,
            'notes': self.notes
        }

class StudentFace(db.Model):
    __tablename__ = 'student_faces'
    
    id = db.Column(db.Integer, primary_key=True)
    student_id = db.Column(db.Integer, db.ForeignKey('students.id'), nullable=False)
    image_path = db.Column(db.String(255), nullable=False)
    face_encoding = db.Column(db.LargeBinary, nullable=False)
    upload_date = db.Column(db.DateTime, default=datetime.utcnow)
    
    student = db.relationship('Student', backref='faces')
    
    def __repr__(self):
        return f'<StudentFace {self.student_id}>'