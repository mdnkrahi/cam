from datetime import datetime, date, timedelta
from models.student import Student, AttendanceRecord
from models import db
from config import get_config
import logging

logger = logging.getLogger(__name__)

class AttendanceManager:
    def __init__(self):
        """Initialize attendance manager"""
        self.config = get_config()
    
    def mark_attendance(self, student_id, confidence_score, status='present', notes=None):
        """
        Mark attendance for a student
        
        Args:
            student_id: Student ID
            confidence_score: Face recognition confidence score
            status: Attendance status (present, late, absent)
            notes: Additional notes
        
        Returns:
            AttendanceRecord object or None if failed
        """
        try:
            student = Student.query.get(student_id)
            if not student:
                logger.error(f"Student {student_id} not found")
                return None
            
            today = date.today()
            
            # Check if already marked today
            existing_record = AttendanceRecord.query.filter_by(
                student_id=student_id,
                date=today
            ).first()
            
            if existing_record:
                # Update check_out time if already checked in
                if not existing_record.check_out_time:
                    existing_record.check_out_time = datetime.now()
                    db.session.commit()
                    logger.info(f"Checked out student {student.name}")
                    return existing_record
                else:
                    logger.info(f"Student {student.name} already marked today")
                    return existing_record
            
            # Create new attendance record
            record = AttendanceRecord(
                student_id=student_id,
                date=today,
                check_in_time=datetime.now(),
                confidence_score=confidence_score,
                status=status,
                notes=notes
            )
            
            db.session.add(record)
            db.session.commit()
            
            logger.info(f"Marked attendance for student {student.name} with confidence {confidence_score:.2f}")
            return record
        
        except Exception as e:
            logger.error(f"Error marking attendance: {str(e)}")
            db.session.rollback()
            return None
    
    def get_attendance_by_date(self, date_obj, class_name=None):
        """
        Get attendance records for a specific date
        
        Args:
            date_obj: Date object
            class_name: Optional class filter
        
        Returns:
            List of attendance records
        """
        try:
            query = AttendanceRecord.query.filter_by(date=date_obj)
            
            if class_name:
                query = query.join(Student).filter(Student.class_name == class_name)
            
            records = query.all()
            return records
        
        except Exception as e:
            logger.error(f"Error fetching attendance records: {str(e)}")
            return []
    
    def get_attendance_by_student(self, student_id, start_date=None, end_date=None):
        """
        Get attendance records for a student within a date range
        
        Args:
            student_id: Student ID
            start_date: Start date (defaults to 30 days ago)
            end_date: End date (defaults to today)
        
        Returns:
            List of attendance records
        """
        try:
            if not end_date:
                end_date = date.today()
            
            if not start_date:
                start_date = end_date - timedelta(days=30)
            
            records = AttendanceRecord.query.filter(
                AttendanceRecord.student_id == student_id,
                AttendanceRecord.date >= start_date,
                AttendanceRecord.date <= end_date
            ).order_by(AttendanceRecord.date.desc()).all()
            
            return records
        
        except Exception as e:
            logger.error(f"Error fetching student attendance: {str(e)}")
            return []
    
    def get_attendance_stats(self, student_id, start_date=None, end_date=None):
        """
        Calculate attendance statistics for a student
        
        Args:
            student_id: Student ID
            start_date: Start date
            end_date: End date
        
        Returns:
            Dictionary with attendance statistics
        """
        try:
            records = self.get_attendance_by_student(student_id, start_date, end_date)
            
            if not records:
                return {
                    'total_days': 0,
                    'present_days': 0,
                    'absent_days': 0,
                    'late_days': 0,
                    'attendance_percentage': 0
                }
            
            total_days = len(records)
            present_days = len([r for r in records if r.status == 'present'])
            late_days = len([r for r in records if r.status == 'late'])
            absent_days = len([r for r in records if r.status == 'absent'])
            
            attendance_percentage = (present_days / total_days * 100) if total_days > 0 else 0
            
            return {
                'total_days': total_days,
                'present_days': present_days,
                'late_days': late_days,
                'absent_days': absent_days,
                'attendance_percentage': round(attendance_percentage, 2)
            }
        
        except Exception as e:
            logger.error(f"Error calculating attendance stats: {str(e)}")
            return None
    
    def get_class_attendance_stats(self, class_name, date_obj=None):
        """
        Get attendance statistics for entire class
        
        Args:
            class_name: Class name
            date_obj: Date (defaults to today)
        
        Returns:
            Dictionary with class statistics
        """
        try:
            if not date_obj:
                date_obj = date.today()
            
            total_students = Student.query.filter_by(class_name=class_name, is_active=True).count()
            
            attendance_records = AttendanceRecord.query.join(Student).filter(
                Student.class_name == class_name,
                AttendanceRecord.date == date_obj
            ).all()
            
            present = len([r for r in attendance_records if r.status == 'present'])
            late = len([r for r in attendance_records if r.status == 'late'])
            absent = total_students - present - late
            
            attendance_percentage = (present / total_students * 100) if total_students > 0 else 0
            
            return {
                'total_students': total_students,
                'present': present,
                'late': late,
                'absent': absent,
                'attendance_percentage': round(attendance_percentage, 2)
            }
        
        except Exception as e:
            logger.error(f"Error calculating class stats: {str(e)}")
            return None
    
    def export_attendance_to_csv(self, class_name=None, start_date=None, end_date=None):
        """
        Export attendance records to CSV format
        
        Args:
            class_name: Class name filter
            start_date: Start date
            end_date: End date
        
        Returns:
            CSV data as string
        """
        try:
            import csv
            from io import StringIO
            
            if not end_date:
                end_date = date.today()
            
            if not start_date:
                start_date = end_date - timedelta(days=30)
            
            # Build query
            query = AttendanceRecord.query.filter(
                AttendanceRecord.date >= start_date,
                AttendanceRecord.date <= end_date
            )
            
            if class_name:
                query = query.join(Student).filter(Student.class_name == class_name)
            else:
                query = query.join(Student)
            
            records = query.order_by(Student.name, AttendanceRecord.date).all()
            
            # Create CSV
            output = StringIO()
            writer = csv.writer(output)
            
            # Headers
            writer.writerow([
                'Student ID', 'Name', 'Class', 'Date',
                'Check-in Time', 'Check-out Time', 'Status', 'Confidence'
            ])
            
            # Data rows
            for record in records:
                writer.writerow([
                    record.student.student_id,
                    record.student.name,
                    record.student.class_name,
                    record.date.strftime('%Y-%m-%d'),
                    record.check_in_time.strftime('%H:%M:%S'),
                    record.check_out_time.strftime('%H:%M:%S') if record.check_out_time else '-',
                    record.status,
                    f"{record.confidence_score:.2f}"
                ])
            
            return output.getvalue()
        
        except Exception as e:
            logger.error(f"Error exporting to CSV: {str(e)}")
            return None