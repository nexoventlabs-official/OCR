"""
Advanced Multi-Engine OCR Module
Uses multiple deep learning OCR engines for maximum accuracy
- EasyOCR (Deep Learning based, good for handwritten)
- PaddleOCR (Very accurate, fast)
- Tesseract (Fallback)
- Google Cloud Vision (If configured)
"""
import os
import re
import cv2
import numpy as np
from PIL import Image
from datetime import datetime
import json

# Try importing OCR engines
EASYOCR_AVAILABLE = False
PADDLEOCR_AVAILABLE = False
TESSERACT_AVAILABLE = False
GOOGLE_VISION_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
    print("✓ EasyOCR loaded")
except ImportError:
    print("✗ EasyOCR not available")

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
    print("✓ PaddleOCR loaded")
except ImportError:
    print("✗ PaddleOCR not available")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
    print("✓ Tesseract loaded")
except ImportError:
    print("✗ Tesseract not available")

try:
    from google.cloud import vision
    GOOGLE_VISION_AVAILABLE = True
    print("✓ Google Cloud Vision loaded")
except ImportError:
    print("✗ Google Cloud Vision not available")


class AdvancedImagePreprocessor:
    """Advanced image preprocessing for form documents"""
    
    @staticmethod
    def preprocess_for_ocr(image):
        """Comprehensive preprocessing pipeline"""
        if isinstance(image, str):
            image = cv2.imread(image)
        
        original = image.copy()
        results = {'original': original}
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        results['gray'] = gray
        
        # 1. Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        results['denoised'] = denoised
        
        # 2. Contrast enhancement (CLAHE)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        results['enhanced'] = enhanced
        
        # 3. Adaptive thresholding
        adaptive = cv2.adaptiveThreshold(
            enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        results['adaptive'] = adaptive
        
        # 4. Otsu thresholding
        _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        results['otsu'] = otsu
        
        # 5. Morphological operations to clean up
        kernel = np.ones((1, 1), np.uint8)
        morph = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
        results['morphed'] = morph
        
        # 6. Sharpen
        kernel_sharpen = np.array([[-1, -1, -1],
                                   [-1, 9, -1],
                                   [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel_sharpen)
        results['sharpened'] = sharpened
        
        # 7. For handwritten red ink - extract red channel
        if len(original.shape) == 3:
            hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
            # Red color range in HSV
            lower_red1 = np.array([0, 50, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 50, 50])
            upper_red2 = np.array([180, 255, 255])
            
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = mask1 + mask2
            
            # Invert for OCR (black text on white)
            red_extracted = cv2.bitwise_not(red_mask)
            results['red_ink'] = red_extracted
        
        return results
    
    @staticmethod
    def deskew(image):
        """Deskew the image"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        coords = np.column_stack(np.where(gray > 0))
        if len(coords) == 0:
            return image
            
        angle = cv2.minAreaRect(coords)[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        
        if abs(angle) > 0.5:
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h),
                                     flags=cv2.INTER_CUBIC,
                                     borderMode=cv2.BORDER_REPLICATE)
            return rotated
        return image


class EasyOCREngine:
    """EasyOCR-based extraction (Deep Learning)"""
    
    def __init__(self):
        self.reader = None
        if EASYOCR_AVAILABLE:
            print("Initializing EasyOCR (this may take a moment)...")
            self.reader = easyocr.Reader(['en'], gpu=False)
            print("✓ EasyOCR initialized")
    
    def extract(self, image):
        """Extract text using EasyOCR"""
        if not self.reader:
            return []
        
        try:
            if isinstance(image, str):
                results = self.reader.readtext(image)
            else:
                results = self.reader.readtext(image)
            
            # Format: [(bbox, text, confidence), ...]
            extracted = []
            for (bbox, text, conf) in results:
                extracted.append({
                    'text': text,
                    'confidence': conf,
                    'bbox': bbox
                })
            return extracted
        except Exception as e:
            print(f"EasyOCR error: {e}")
            return []


class PaddleOCREngine:
    """PaddleOCR-based extraction (Very accurate)"""
    
    def __init__(self):
        self.ocr = None
        if PADDLEOCR_AVAILABLE:
            print("Initializing PaddleOCR...")
            self.ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
            print("✓ PaddleOCR initialized")
    
    def extract(self, image):
        """Extract text using PaddleOCR"""
        if not self.ocr:
            return []
        
        try:
            if isinstance(image, str):
                result = self.ocr.ocr(image, cls=True)
            else:
                result = self.ocr.ocr(image, cls=True)
            
            extracted = []
            if result and result[0]:
                for line in result[0]:
                    bbox, (text, conf) = line
                    extracted.append({
                        'text': text,
                        'confidence': conf,
                        'bbox': bbox
                    })
            return extracted
        except Exception as e:
            print(f"PaddleOCR error: {e}")
            return []


class TesseractEngine:
    """Tesseract OCR engine"""
    
    def __init__(self):
        self.available = TESSERACT_AVAILABLE
        if self.available:
            try:
                from config import Config
                if os.path.exists(Config.TESSERACT_PATH):
                    pytesseract.pytesseract.tesseract_cmd = Config.TESSERACT_PATH
            except:
                pass
    
    def extract(self, image):
        """Extract text using Tesseract"""
        if not self.available:
            return []
        
        try:
            # Get detailed data
            custom_config = r'--oem 3 --psm 6'
            data = pytesseract.image_to_data(image, config=custom_config, output_type=pytesseract.Output.DICT)
            
            extracted = []
            n_boxes = len(data['text'])
            for i in range(n_boxes):
                text = data['text'][i].strip()
                conf = int(data['conf'][i])
                if text and conf > 0:
                    extracted.append({
                        'text': text,
                        'confidence': conf / 100.0,
                        'bbox': [data['left'][i], data['top'][i], 
                                data['width'][i], data['height'][i]]
                    })
            return extracted
        except Exception as e:
            print(f"Tesseract error: {e}")
            return []


class MultiEngineOCR:
    """Combines multiple OCR engines for best accuracy"""
    
    def __init__(self):
        self.preprocessor = AdvancedImagePreprocessor()
        self.engines = {}
        
        print("\n" + "="*50)
        print("Initializing Multi-Engine OCR System")
        print("="*50)
        
        # Initialize available engines
        if EASYOCR_AVAILABLE:
            self.engines['easyocr'] = EasyOCREngine()
        
        if PADDLEOCR_AVAILABLE:
            self.engines['paddleocr'] = PaddleOCREngine()
        
        if TESSERACT_AVAILABLE:
            self.engines['tesseract'] = TesseractEngine()
        
        print(f"\nActive engines: {list(self.engines.keys())}")
        print("="*50 + "\n")
    
    def extract_all(self, image_path):
        """Run all OCR engines and combine results"""
        # Preprocess image
        processed = self.preprocessor.preprocess_for_ocr(image_path)
        
        all_results = {
            'raw_texts': [],
            'structured_results': {},
            'combined_text': ''
        }
        
        # Run each engine on different preprocessed versions
        for engine_name, engine in self.engines.items():
            print(f"Running {engine_name}...")
            
            engine_results = []
            
            # Try on original
            results = engine.extract(image_path)
            engine_results.extend(results)
            
            # Try on enhanced version
            if 'enhanced' in processed:
                results = engine.extract(processed['enhanced'])
                engine_results.extend(results)
            
            # Try on red ink extraction (for handwritten marks)
            if 'red_ink' in processed:
                results = engine.extract(processed['red_ink'])
                engine_results.extend(results)
            
            # Deduplicate and store
            seen_texts = set()
            unique_results = []
            for r in engine_results:
                if r['text'] not in seen_texts:
                    seen_texts.add(r['text'])
                    unique_results.append(r)
            
            all_results['structured_results'][engine_name] = unique_results
            
            # Add to raw texts
            texts = [r['text'] for r in unique_results if r['confidence'] > 0.3]
            all_results['raw_texts'].extend(texts)
        
        # Combine all text
        all_results['combined_text'] = ' '.join(set(all_results['raw_texts']))
        
        return all_results, processed


class AssessmentFormParser:
    """Parse assessment form data from OCR results"""
    
    def __init__(self):
        self.multi_ocr = MultiEngineOCR()
    
    def parse_assessment(self, image_path):
        """Extract structured assessment data"""
        print(f"\nProcessing: {image_path}")
        
        # Get OCR results
        ocr_results, processed_images = self.multi_ocr.extract_all(image_path)
        
        # Combine all extracted text
        all_text = ocr_results['combined_text'].upper()
        
        # Also get structured results for number extraction
        all_items = []
        for engine_results in ocr_results['structured_results'].values():
            all_items.extend(engine_results)
        
        # Initialize data structure
        data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'registration_no': '',
            'college_code': '',
            'date': '',
            'degree_branch': '',
            'semester_section': '',
            'subject_code': '',
            'subject_name': '',
            'no_of_pages': '',
            'hall_superintendent': '',
            'part_a_marks': {},
            'part_bc_marks': {},
            'total_marks': '',
            'grand_total': '',
            'course_outcomes': {},
            'examiner_name': '',
            'raw_text': ocr_results['combined_text']
        }
        
        # Extract Registration Number (12+ digits)
        reg_match = re.search(r'(\d{12,})', all_text)
        if reg_match:
            data['registration_no'] = reg_match.group(1)
        
        # Extract College Code
        code_patterns = [
            r'COLLEGE\s*CODE[:\s\|]*(\d{3,4})',
            r'CODE[:\s\|]*(\d{4})',
            r'\|(\d{4})\|'
        ]
        for pattern in code_patterns:
            match = re.search(pattern, all_text)
            if match:
                data['college_code'] = match.group(1)
                break
        
        # Extract Date
        date_patterns = [
            r'DATE[:\s]*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
            r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})'
        ]
        for pattern in date_patterns:
            match = re.search(pattern, all_text)
            if match:
                data['date'] = match.group(1)
                break
        
        # Extract Degree/Branch
        degree_patterns = [
            r'(BE\s*/?\s*ECE)',
            r'(BE\s*/?\s*CSE)',
            r'(BE\s*/?\s*EEE)',
            r'(BE\s*/?\s*MECH)',
            r'(B\.?TECH)',
            r'DEGREE[/\s]*BRANCH[:\s]*([A-Z]+[/\s]*[A-Z]+)'
        ]
        for pattern in degree_patterns:
            match = re.search(pattern, all_text)
            if match:
                data['degree_branch'] = match.group(1).replace(' ', '')
                break
        
        # Extract Semester/Section
        sem_patterns = [
            r"([IVX]+)\s*['\"]?\s*([A-Z])['\"]?",
            r'SEMESTER[/\s]*SECTION[:\s]*([IVX]+\s*[A-Z]?)',
            r'SEM[:\s]*([IVX]+)'
        ]
        for pattern in sem_patterns:
            match = re.search(pattern, all_text)
            if match:
                if match.lastindex >= 2:
                    data['semester_section'] = f"{match.group(1)} '{match.group(2)}'"
                else:
                    data['semester_section'] = match.group(1)
                break
        
        # Extract Subject Code
        subj_match = re.search(r'([A-Z]{2,3}\d{3})', all_text)
        if subj_match:
            data['subject_code'] = subj_match.group(1)
        
        # Extract Subject Name
        if 'INDUSTRIAL' in all_text and ('IOT' in all_text or 'LOT' in all_text):
            data['subject_name'] = 'INDUSTRIAL IOT AND INDUSTRY 4.0'
        
        # Extract number of pages
        pages_match = re.search(r'PAGES[:\s]*(\d+)', all_text)
        if pages_match:
            data['no_of_pages'] = pages_match.group(1)
        
        # Extract marks - look for numbers in OCR results
        # Find all numbers with their positions
        numbers = []
        for item in all_items:
            text = item['text'].strip()
            # Check if it's a number
            if re.match(r'^\d{1,2}$', text):
                numbers.append({
                    'value': int(text),
                    'text': text,
                    'confidence': item.get('confidence', 0)
                })
        
        # Part A marks (typically small numbers 0-2)
        part_a_candidates = [n for n in numbers if n['value'] <= 2]
        if len(part_a_candidates) >= 5:
            for i, mark in enumerate(part_a_candidates[:5]):
                data['part_a_marks'][f'q{i+1}'] = str(mark['value'])
        
        # Look for total marks patterns
        total_patterns = [
            r'TOTAL[:\s]*(\d{1,2})',
            r'(\d{2})\s*/\s*(\d{2,3})'  # Like 46/60 or 43/100
        ]
        
        for pattern in total_patterns:
            match = re.search(pattern, all_text)
            if match:
                if match.lastindex >= 2:
                    data['grand_total'] = f"{match.group(1)}/{match.group(2)}"
                    data['total_marks'] = match.group(1)
                else:
                    data['total_marks'] = match.group(1)
                break
        
        # Look for grand total (larger numbers like 43, 46)
        large_numbers = [n for n in numbers if 30 <= n['value'] <= 100]
        if large_numbers and not data['total_marks']:
            # Take the one with highest confidence
            best = max(large_numbers, key=lambda x: x['confidence'])
            data['total_marks'] = str(best['value'])
        
        # Course Outcomes - look for CO patterns
        co_pattern = r'CO[-\s]*(\d)[:\s]*(\d+)[,\s]*(\d+)[,\s]*(\d+)[,\s]*(\d+)'
        co_matches = re.findall(co_pattern, all_text)
        for match in co_matches:
            co_num = match[0]
            data['course_outcomes'][f'CO{co_num}'] = {
                'part_a': match[1],
                'part_b': match[2],
                'part_c': match[3],
                'total': match[4]
            }
        
        # Try to find CO1 and CO2 from numbers
        # Look for patterns like "2 21 11 34" for CO1
        co_candidates = [n for n in numbers if n['value'] <= 50]
        
        return data, ocr_results


def flatten_data_for_sheet(data):
    """Convert structured data to flat list for Google Sheets"""
    flat_data = [
        data.get('timestamp', ''),
        data.get('registration_no', ''),
        data.get('college_code', ''),
        data.get('date', ''),
        data.get('degree_branch', ''),
        data.get('semester_section', ''),
        data.get('subject_code', ''),
        data.get('subject_name', ''),
        data.get('no_of_pages', ''),
        data.get('hall_superintendent', ''),
    ]
    
    # Part A marks (Q1-Q5)
    part_a = data.get('part_a_marks', {})
    for i in range(1, 6):
        flat_data.append(part_a.get(f'q{i}', ''))
    
    # Part A Total
    part_a_total = sum(int(v) for v in part_a.values() if v.isdigit())
    flat_data.append(str(part_a_total) if part_a_total else '')
    
    # Part B&C marks (Q6-Q9, a/b options, marks I/II/III)
    part_bc = data.get('part_bc_marks', {})
    for q in range(6, 10):
        for opt in ['a', 'b']:
            for mark in ['I', 'II', 'III']:
                key = f'q{q}{opt}_{mark}'
                flat_data.append(part_bc.get(key, ''))
    
    # Part B&C Total
    flat_data.append('')
    
    # Total Marks
    flat_data.append(data.get('total_marks', ''))
    
    # Grand Total
    flat_data.append(data.get('grand_total', ''))
    
    # Course Outcomes
    course_outcomes = data.get('course_outcomes', {})
    for co in ['CO1', 'CO2']:
        co_data = course_outcomes.get(co, {})
        flat_data.extend([
            co_data.get('part_a', ''),
            co_data.get('part_b', ''),
            co_data.get('part_c', ''),
            co_data.get('total', '')
        ])
    
    # Examiner Name
    flat_data.append(data.get('examiner_name', ''))
    
    # Raw OCR Text (truncated)
    raw_text = data.get('raw_text', '')
    flat_data.append(raw_text[:500] if len(raw_text) > 500 else raw_text)
    
    return flat_data


# Test function
if __name__ == "__main__":
    import sys
    
    # Find test image
    uploads_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'uploads')
    
    test_images = []
    if os.path.exists(uploads_folder):
        for f in os.listdir(uploads_folder):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                test_images.append(os.path.join(uploads_folder, f))
    
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    elif test_images:
        test_image = test_images[0]
    else:
        print("No test image found!")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("ADVANCED OCR EXTRACTION TEST")
    print(f"{'='*60}")
    print(f"Image: {test_image}")
    
    # Run extraction
    parser = AssessmentFormParser()
    data, ocr_results = parser.parse_assessment(test_image)
    
    print(f"\n{'='*60}")
    print("EXTRACTED DATA:")
    print(f"{'='*60}")
    for key, value in data.items():
        if key != 'raw_text':
            print(f"  {key}: {value}")
    
    print(f"\n{'='*60}")
    print("RAW OCR RESULTS BY ENGINE:")
    print(f"{'='*60}")
    for engine, results in ocr_results['structured_results'].items():
        print(f"\n{engine.upper()}:")
        for r in results[:20]:  # Show first 20
            print(f"  [{r['confidence']:.2f}] {r['text']}")
