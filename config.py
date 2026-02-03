"""
Configuration settings for the OCR Assessment Scanner
"""
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration class"""
    
    # Google Sheets Configuration
    GOOGLE_SERVICE_ACCOUNT_KEY = os.getenv('GOOGLE_SERVICE_ACCOUNT_KEY', '{}')
    GOOGLE_CREDENTIALS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'credentials.json')
    GOOGLE_SHEET_ID = os.getenv('GOOGLE_SHEET_ID', '1J-CgWOa9ZDy4jeqInrw4BMlXoPXqD-_F6Xokw0igFyQ')
    
    # Tesseract Configuration
    TESSERACT_PATH = os.getenv('TESSERACT_PATH', r'C:\Program Files\Tesseract-OCR\tesseract.exe')
    
    # Groq Configuration (for AI-powered extraction)
    GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')
    
    # Cloudinary Configuration (Cloud Image Storage)
    CLOUDINARY_CLOUD_NAME = os.getenv('CLOUDINARY_CLOUD_NAME', 'dlzdz5alh')
    CLOUDINARY_API_KEY = os.getenv('CLOUDINARY_API_KEY', '682695357187379')
    CLOUDINARY_API_SECRET = os.getenv('CLOUDINARY_API_SECRET', '_c3twlsJrga3sNLtZpcapicN9kA')
    USE_CLOUDINARY = os.getenv('USE_CLOUDINARY', 'true').lower() == 'true'
    
    # Application Settings
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Webcam Settings
    CAMERA_INDEX = 0
    CAPTURE_WIDTH = 1280
    CAPTURE_HEIGHT = 720
    
    # Real-Time OCR Settings
    REALTIME_OCR_VIEW_MODE = 1  # 1=high conf, 2=color code, 3=gradient, 4=all
    REALTIME_OCR_CROP = (100, 100)  # (width_crop, height_crop) in pixels
    REALTIME_OCR_LANGUAGE = 'eng'  # Tesseract language code
    REALTIME_OCR_USE_EASYOCR = False  # Use EasyOCR instead of Tesseract
    
    @classmethod
    def get_google_credentials(cls):
        """Parse and return Google service account credentials"""
        # First try to load from credentials.json file
        if os.path.exists(cls.GOOGLE_CREDENTIALS_FILE):
            try:
                with open(cls.GOOGLE_CREDENTIALS_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading credentials file: {e}")
        
        # Fall back to environment variable
        try:
            return json.loads(cls.GOOGLE_SERVICE_ACCOUNT_KEY)
        except json.JSONDecodeError:
            return {}
    
    @classmethod
    def ensure_upload_folder(cls):
        """Create upload folder if it doesn't exist"""
        if not os.path.exists(cls.UPLOAD_FOLDER):
            os.makedirs(cls.UPLOAD_FOLDER)
        return cls.UPLOAD_FOLDER


# Sheet column headers for the assessment data
SHEET_HEADERS = [
    'Timestamp',
    'Registration No',
    'College Code',
    'Date',
    'Degree/Branch',
    'Semester/Section',
    'Subject Code',
    'Subject Name',
    'No of Pages',
    'Hall Superintendent',
    'Part A Q1 Marks',
    'Part A Q2 Marks',
    'Part A Q3 Marks',
    'Part A Q4 Marks',
    'Part A Q5 Marks',
    'Part A Total',
    'Part B Q6a Marks I',
    'Part B Q6a Marks II',
    'Part B Q6a Marks III',
    'Part B Q6b Marks I',
    'Part B Q6b Marks II',
    'Part B Q6b Marks III',
    'Part B Q7a Marks I',
    'Part B Q7a Marks II',
    'Part B Q7a Marks III',
    'Part B Q7b Marks I',
    'Part B Q7b Marks II',
    'Part B Q7b Marks III',
    'Part B Q8a Marks I',
    'Part B Q8a Marks II',
    'Part B Q8a Marks III',
    'Part B Q8b Marks I',
    'Part B Q8b Marks II',
    'Part B Q8b Marks III',
    'Part B Q9a Marks I',
    'Part B Q9a Marks II',
    'Part B Q9a Marks III',
    'Part B Q9b Marks I',
    'Part B Q9b Marks II',
    'Part B Q9b Marks III',
    'Part B&C Total',
    'Total Marks',
    'Grand Total (out of 100)',
    'CO1 Part A',
    'CO1 Part B',
    'CO1 Part C',
    'CO1 Total',
    'CO2 Part A',
    'CO2 Part B',
    'CO2 Part C',
    'CO2 Total',
    'Examiner Name',
    'Raw OCR Text'
]
