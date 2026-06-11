import os
from datetime import timedelta

# Flask Configuration
class Config:
    # Application settings
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')
    DEBUG = os.environ.get('DEBUG', True)
    
    # Database
    SQLALCHEMY_DATABASE_URI = os.environ.get('DATABASE_URL', 'sqlite:///attendance.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # YOLO Configuration
    YOLO_MODEL = 'yolov8n-face.pt'  # nano model for face detection
    CONFIDENCE_THRESHOLD = 0.5
    MAX_DETECTIONS = 10
    
    # Face Recognition
    FACE_RECOGNITION_THRESHOLD = 0.6
    MAX_FACES_PER_STUDENT = 5
    
    # File Upload
    UPLOAD_FOLDER = 'data/student_faces'
    ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Attendance Rules
    ATTENDANCE_TIMEOUT = timedelta(hours=24)  # Mark attendance once per 24 hours
    ATTENDANCE_CONFIDENCE = 0.8  # Confidence threshold for marking attendance
    
    # Camera Settings
    CAMERA_INDEX = 0  # Default camera
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    FPS = 30
    
    # Logging
    LOG_FILE = 'logs/attendance.log'
    LOG_LEVEL = 'INFO'
    
    # Session timeout (minutes)
    PERMANENT_SESSION_LIFETIME = 30

# Development Configuration
class DevelopmentConfig(Config):
    DEBUG = True
    TESTING = False

# Production Configuration
class ProductionConfig(Config):
    DEBUG = False
    TESTING = False

# Testing Configuration
class TestingConfig(Config):
    DEBUG = True
    TESTING = True
    SQLALCHEMY_DATABASE_URI = 'sqlite:///:memory:'

# Select configuration based on environment
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}

def get_config():
    env = os.environ.get('FLASK_ENV', 'development')
    return config.get(env, config['default'])