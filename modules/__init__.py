"""
Modules package initialization
"""
from .ocr_extractor import AssessmentDataExtractor, flatten_data_for_sheet
from .sheets_manager import GoogleSheetsManager, get_sheets_manager
from .webcam_capture import WebcamManager, ImageCapture, get_webcam_manager

# Try to import advanced OCR
try:
    from .advanced_ocr import AssessmentFormParser, MultiEngineOCR
    ADVANCED_OCR_AVAILABLE = True
except ImportError:
    ADVANCED_OCR_AVAILABLE = False

__all__ = [
    'AssessmentDataExtractor',
    'flatten_data_for_sheet',
    'GoogleSheetsManager',
    'get_sheets_manager',
    'WebcamManager',
    'ImageCapture',
    'get_webcam_manager',
    'AssessmentFormParser',
    'MultiEngineOCR',
    'ADVANCED_OCR_AVAILABLE'
]
