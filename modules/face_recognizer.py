import cv2
import face_recognition
import numpy as np
from models.student import Student, StudentFace
from models import db
from config import get_config
import logging
import pickle
import os

logger = logging.getLogger(__name__)

class FaceRecognizer:
    def __init__(self):
        """Initialize face recognizer"""
        self.config = get_config()
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        self.load_known_faces()
    
    def encode_face(self, face_image):
        """
        Encode a face image to embedding
        
        Args:
            face_image: Face image (RGB format)
        
        Returns:
            Face encoding or None if no face detected
        """
        try:
            # Resize for faster processing
            small_frame = cv2.resize(face_image, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Find faces
            face_locations = face_recognition.face_locations(rgb_small_frame)
            
            if len(face_locations) == 0:
                return None
            
            # Encode the first face found
            face_encoding = face_recognition.face_encodings(
                rgb_small_frame, 
                face_locations
            )[0]
            
            return face_encoding
        
        except Exception as e:
            logger.error(f"Error encoding face: {str(e)}")
            return None
    
    def load_known_faces(self):
        """Load all known student faces from database"""
        try:
            students = Student.query.filter_by(is_active=True).all()
            
            for student in students:
                if student.face_encoding:
                    try:
                        encoding = pickle.loads(student.face_encoding)
                        self.known_face_encodings.append(encoding)
                        self.known_face_names.append(student.name)
                        self.known_face_ids.append(student.id)
                    except Exception as e:
                        logger.warning(f"Error loading encoding for {student.name}: {str(e)}")
            
            logger.info(f"Loaded {len(self.known_face_encodings)} known face encodings")
        
        except Exception as e:
            logger.error(f"Error loading known faces: {str(e)}")
    
    def recognize_faces(self, face_image, tolerance=None):
        """
        Recognize faces in an image
        
        Args:
            face_image: Face image (RGB format)
            tolerance: Recognition tolerance (uses config default if None)
        
        Returns:
            List of recognized faces with IDs and confidence
        """
        if tolerance is None:
            tolerance = 1 - self.config.FACE_RECOGNITION_THRESHOLD
        
        try:
            if len(self.known_face_encodings) == 0:
                logger.warning("No known faces loaded")
                return []
            
            # Encode the input face
            test_encoding = self.encode_face(face_image)
            
            if test_encoding is None:
                return []
            
            # Compare with known faces
            matches = face_recognition.compare_faces(
                self.known_face_encodings,
                test_encoding,
                tolerance=tolerance
            )
            
            # Calculate face distances
            face_distances = face_recognition.face_distance(
                self.known_face_encodings,
                test_encoding
            )
            
            results = []
            
            for i, match in enumerate(matches):
                if match:
                    distance = face_distances[i]
                    confidence = 1 - distance
                    
                    if confidence >= self.config.FACE_RECOGNITION_THRESHOLD:
                        results.append({
                            'student_id': self.known_face_ids[i],
                            'name': self.known_face_names[i],
                            'confidence': float(confidence),
                            'distance': float(distance)
                        })
            
            # Sort by confidence
            results.sort(key=lambda x: x['confidence'], reverse=True)
            
            return results
        
        except Exception as e:
            logger.error(f"Error recognizing faces: {str(e)}")
            return []
    
    def register_student_face(self, student_id, face_image):
        """
        Register a new student face
        
        Args:
            student_id: Student ID
            face_image: Face image (BGR format)
        
        Returns:
            Success status
        """
        try:
            student = Student.query.get(student_id)
            if not student:
                logger.error(f"Student {student_id} not found")
                return False
            
            # Encode face
            face_encoding = self.encode_face(face_image)
            
            if face_encoding is None:
                logger.error("Could not encode face")
                return False
            
            # Store in database
            student.face_encoding = pickle.dumps(face_encoding)
            db.session.commit()
            
            # Update in-memory database
            if student.id in self.known_face_ids:
                idx = self.known_face_ids.index(student.id)
                self.known_face_encodings[idx] = face_encoding
                self.known_face_names[idx] = student.name
            else:
                self.known_face_encodings.append(face_encoding)
                self.known_face_names.append(student.name)
                self.known_face_ids.append(student.id)
            
            logger.info(f"Registered face for student {student.name}")
            return True
        
        except Exception as e:
            logger.error(f"Error registering student face: {str(e)}")
            return False
    
    def refresh_known_faces(self):
        """Refresh the known faces database"""
        self.known_face_encodings = []
        self.known_face_names = []
        self.known_face_ids = []
        self.load_known_faces()