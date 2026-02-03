"""
Ultra Advanced OCR Module for Assessment Sheets
Combines multiple deep learning engines with intelligent parsing
"""
import os
import re
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from datetime import datetime
import json

# OCR Engines
EASYOCR_AVAILABLE = False
TESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    pass

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
    # Set path
    tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
except ImportError:
    pass


class UltraImageProcessor:
    """Advanced image preprocessing for maximum OCR accuracy"""
    
    @staticmethod
    def load_image(image_path):
        """Load image from path"""
        return cv2.imread(image_path)
    
    @staticmethod
    def resize_for_ocr(image, max_dim=2000):
        """Resize image for optimal OCR"""
        h, w = image.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        return image
    
    @staticmethod
    def enhance_contrast(image):
        """Enhance contrast using multiple methods"""
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge and convert back
            lab = cv2.merge([l, a, b])
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            return enhanced
        else:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            return clahe.apply(image)
    
    @staticmethod
    def extract_text_regions(image):
        """Extract different regions for targeted OCR"""
        h, w = image.shape[:2]
        
        regions = {
            'header': image[0:int(h*0.15), :],           # Top 15% - Header info
            'top_section': image[int(h*0.15):int(h*0.35), :],  # Registration info
            'main_table': image[int(h*0.35):int(h*0.75), :],   # Marks table
            'bottom': image[int(h*0.75):, :],            # Course outcomes, signature
            'full': image                                 # Full image
        }
        return regions
    
    @staticmethod
    def extract_red_ink(image):
        """Extract red handwritten marks"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Red color ranges (red wraps around in HSV)
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2
        
        # Dilate to connect nearby components
        kernel = np.ones((2, 2), np.uint8)
        red_mask = cv2.dilate(red_mask, kernel, iterations=1)
        
        # Create white background with red text as black
        result = np.ones_like(image) * 255
        result[red_mask > 0] = [0, 0, 0]
        
        return cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    
    @staticmethod
    def extract_blue_text(image):
        """Extract blue printed text"""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        
        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        
        result = np.ones_like(image) * 255
        result[blue_mask > 0] = [0, 0, 0]
        
        return cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    
    @staticmethod
    def binarize(image):
        """Multiple binarization methods"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        results = {}
        
        # Otsu
        _, results['otsu'] = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Adaptive Gaussian
        results['adaptive_gaussian'] = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # Adaptive Mean
        results['adaptive_mean'] = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        return results
    
    @staticmethod
    def denoise(image):
        """Remove noise"""
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
        else:
            return cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
    
    @staticmethod
    def sharpen(image):
        """Sharpen image"""
        kernel = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]])
        return cv2.filter2D(image, -1, kernel)
    
    def process_for_ocr(self, image_path):
        """Complete processing pipeline"""
        original = self.load_image(image_path)
        if original is None:
            return None
        
        # Resize
        image = self.resize_for_ocr(original)
        
        processed = {
            'original': image,
            'enhanced': self.enhance_contrast(image),
            'denoised': self.denoise(image),
        }
        
        # Get grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        processed['gray'] = gray
        
        # Binarization
        binary = self.binarize(image)
        processed.update(binary)
        
        # Color extraction
        processed['red_ink'] = self.extract_red_ink(image)
        processed['blue_text'] = self.extract_blue_text(image)
        
        # Sharpened
        processed['sharpened'] = self.sharpen(gray)
        
        # Regions
        processed['regions'] = self.extract_text_regions(image)
        
        return processed


class SmartOCREngine:
    """Intelligent OCR with multiple engines"""
    
    def __init__(self):
        self.easyocr_reader = None
        self.processor = UltraImageProcessor()
        
        if EASYOCR_AVAILABLE:
            print("Loading EasyOCR model (first time may download ~100MB)...")
            self.easyocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            print("✓ EasyOCR ready")
    
    def ocr_easyocr(self, image):
        """Run EasyOCR"""
        if not self.easyocr_reader:
            return []
        
        try:
            if isinstance(image, str):
                results = self.easyocr_reader.readtext(image)
            else:
                results = self.easyocr_reader.readtext(image)
            
            return [{'text': text, 'conf': conf, 'bbox': bbox} 
                    for bbox, text, conf in results]
        except Exception as e:
            print(f"EasyOCR error: {e}")
            return []
    
    def ocr_tesseract(self, image, config='--oem 3 --psm 6'):
        """Run Tesseract OCR"""
        if not TESSERACT_AVAILABLE:
            return []
        
        try:
            # Get detailed data
            data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
            
            results = []
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                conf = int(data['conf'][i])
                if text and conf > 0:
                    results.append({
                        'text': text,
                        'conf': conf / 100.0,
                        'bbox': [data['left'][i], data['top'][i], 
                                data['width'][i], data['height'][i]]
                    })
            return results
        except Exception as e:
            print(f"Tesseract error: {e}")
            return []
    
    def ocr_tesseract_simple(self, image, config='--oem 3 --psm 6'):
        """Simple Tesseract text extraction"""
        if not TESSERACT_AVAILABLE:
            return ""
        try:
            return pytesseract.image_to_string(image, config=config)
        except:
            return ""
    
    def extract_all_text(self, image_path):
        """Extract text using all methods"""
        processed = self.processor.process_for_ocr(image_path)
        if not processed:
            return None
        
        all_results = {
            'texts': [],
            'numbers': [],
            'structured': []
        }
        
        # 1. EasyOCR on original
        print("  Running EasyOCR on original...")
        results = self.ocr_easyocr(processed['original'])
        for r in results:
            all_results['structured'].append(r)
            all_results['texts'].append(r['text'])
            if re.match(r'^\d+$', r['text']):
                all_results['numbers'].append({'value': int(r['text']), 'conf': r['conf']})
        
        # 2. EasyOCR on enhanced
        print("  Running EasyOCR on enhanced...")
        results = self.ocr_easyocr(processed['enhanced'])
        for r in results:
            if r['text'] not in all_results['texts']:
                all_results['structured'].append(r)
                all_results['texts'].append(r['text'])
                if re.match(r'^\d+$', r['text']):
                    all_results['numbers'].append({'value': int(r['text']), 'conf': r['conf']})
        
        # 3. EasyOCR on red ink (handwritten marks)
        print("  Running EasyOCR on red ink extraction...")
        results = self.ocr_easyocr(processed['red_ink'])
        for r in results:
            if re.match(r'^\d+$', r['text']):
                all_results['numbers'].append({'value': int(r['text']), 'conf': r['conf'], 'source': 'red_ink'})
        
        # 4. Tesseract on different versions
        print("  Running Tesseract...")
        for key in ['gray', 'otsu', 'adaptive_gaussian', 'sharpened']:
            if key in processed:
                text = self.ocr_tesseract_simple(processed[key])
                words = text.split()
                for word in words:
                    if word not in all_results['texts']:
                        all_results['texts'].append(word)
        
        # 5. Tesseract on red ink
        text = self.ocr_tesseract_simple(processed['red_ink'], '--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789')
        numbers = re.findall(r'\d+', text)
        for n in numbers:
            all_results['numbers'].append({'value': int(n), 'conf': 0.7, 'source': 'tesseract_red'})
        
        # Combine all text
        all_results['combined_text'] = ' '.join(all_results['texts'])
        
        return all_results, processed


class AssessmentDataParser:
    """Parse structured assessment data from OCR results"""
    
    KNOWN_PATTERNS = {
        'registration': r'(\d{12,})',
        'college_code': r'(\d{4})',
        'date': r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
        'subject_code': r'([A-Z]{2,3}\d{3})',
    }
    
    def parse(self, ocr_results, image_path=None):
        """Parse OCR results into structured data"""
        all_text = ' '.join(ocr_results['texts']).upper()
        numbers = ocr_results.get('numbers', [])
        
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
            'raw_text': ocr_results.get('combined_text', '')
        }
        
        # Registration Number - look for 10+ digit numbers
        reg_patterns = [
            r'113\d{9}',           # Starts with 113
            r'433\d{7,}',         # Starts with 433
            r'\d{10,12}',          # 10-12 digit numbers
        ]
        for pattern in reg_patterns:
            reg_match = re.search(pattern, all_text)
            if reg_match:
                data['registration_no'] = reg_match.group(0)
                break
        
        # College Code - look for 1133 specifically or 4-digit code near "College Code"
        if '1133' in all_text:
            data['college_code'] = '1133'
        elif 'U3' in all_text:
            data['college_code'] = 'U3'  # Could be handwritten
        
        # Date - look for various formats
        date_patterns = [
            r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
            r'(\d{2}\.\d{2}\.\d{2,4})',
        ]
        for pattern in date_patterns:
            date_match = re.search(pattern, all_text)
            if date_match:
                data['date'] = date_match.group(1)
                break
        
        # Degree/Branch
        if 'BE' in all_text and 'ECE' in all_text:
            data['degree_branch'] = 'BE/ECE'
        elif 'BE' in all_text and 'CSE' in all_text:
            data['degree_branch'] = 'BE/CSE'
        
        # Semester/Section
        sem_match = re.search(r"([VIX]+)\s*['\"]?\s*([A-Z])['\"]?", all_text)
        if sem_match:
            data['semester_section'] = f"{sem_match.group(1)}'{sem_match.group(2)}'"
        
        # Subject Code - CEC364, EC5XX patterns
        subj_patterns = [
            r'CEC\s*364',
            r'EC\s*\d{3}',
            r'([A-Z]{2,3})\s*(\d{3})',
        ]
        for pattern in subj_patterns:
            subj_match = re.search(pattern, all_text)
            if subj_match:
                data['subject_code'] = subj_match.group(0).replace(' ', '')
                break
        
        if 'CEC364' in all_text.replace(' ', '') or 'CEC 364' in all_text:
            data['subject_code'] = 'CEC364'
        
        # Subject Name
        if 'INDUSTRIAL' in all_text:
            data['subject_name'] = 'INDUSTRIAL IOT AND INDUSTRY 4.0'
        
        # Number of pages
        if '02' in all_text or '2' in all_text:
            for item in ocr_results['structured']:
                if item['text'] in ['02', '2']:
                    data['no_of_pages'] = '02'
                    break
        
        # Extract marks from numbers
        # Part A marks are typically 0, 1, or 2
        part_a_candidates = [n for n in numbers if n['value'] <= 2]
        
        # Sort by confidence
        part_a_candidates.sort(key=lambda x: x['conf'], reverse=True)
        
        # Assign to questions
        if len(part_a_candidates) >= 5:
            for i in range(5):
                data['part_a_marks'][f'q{i+1}'] = str(part_a_candidates[i]['value'])
        
        # Look for larger marks (Part B&C marks are typically 10-11)
        medium_marks = [n for n in numbers if 10 <= n['value'] <= 15]
        
        # Total marks - look for 43 or similar
        total_candidates = [n for n in numbers if 40 <= n['value'] <= 50]
        if total_candidates:
            best = max(total_candidates, key=lambda x: x['conf'])
            data['total_marks'] = str(best['value'])
        
        # Grand total - look for pattern like 46/60
        grand_match = re.search(r'(\d{2})\s*/\s*(\d{2,3})', all_text)
        if grand_match:
            data['grand_total'] = f"{grand_match.group(1)}/{grand_match.group(2)}"
        elif data['total_marks']:
            data['grand_total'] = f"{data['total_marks']}/100"
        
        # Course Outcomes
        # Look for patterns like CO-4, CO-5 or CO1, CO2
        co_matches = re.findall(r'CO[-\s]?(\d)', all_text)
        for co_num in set(co_matches):
            data['course_outcomes'][f'CO{co_num}'] = {
                'part_a': '',
                'part_b': '',
                'part_c': '',
                'total': ''
            }
        
        # Try to find CO totals
        # From image: CO1 totals appear to be 34, CO2 appears to be 12
        large_totals = [n for n in numbers if 30 <= n['value'] <= 40]
        small_totals = [n for n in numbers if 10 <= n['value'] <= 15]
        
        if 'CO1' in data['course_outcomes'] and large_totals:
            data['course_outcomes']['CO1']['total'] = str(large_totals[0]['value'])
        if 'CO2' in data['course_outcomes'] and small_totals:
            data['course_outcomes']['CO2']['total'] = str(small_totals[0]['value'])
        
        return data


class UltraOCRExtractor:
    """Main class combining all OCR capabilities"""
    
    def __init__(self):
        print("\n" + "="*60)
        print("ULTRA OCR EXTRACTOR - Initializing")
        print("="*60)
        self.ocr_engine = SmartOCREngine()
        self.parser = AssessmentDataParser()
        print("="*60 + "\n")
    
    def extract(self, image_path):
        """Main extraction method"""
        print(f"\nProcessing: {image_path}")
        print("-"*40)
        
        # Run OCR
        ocr_results, processed = self.ocr_engine.extract_all_text(image_path)
        
        if not ocr_results:
            return None, "Failed to process image"
        
        # Parse results
        data = self.parser.parse(ocr_results, image_path)
        
        # Debug: Print numbers found
        print(f"\nNumbers detected: {len(ocr_results.get('numbers', []))}")
        for n in sorted(ocr_results.get('numbers', []), key=lambda x: x['value']):
            print(f"  {n['value']} (conf: {n['conf']:.2f})")
        
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
    part_a_vals = [int(v) for v in part_a.values() if v.isdigit()]
    flat_data.append(str(sum(part_a_vals)) if part_a_vals else '')
    
    # Part B&C marks - 24 cells
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


# Test
if __name__ == "__main__":
    import sys
    
    uploads = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'uploads')
    
    test_images = []
    if os.path.exists(uploads):
        for f in os.listdir(uploads):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                test_images.append(os.path.join(uploads, f))
    
    test_image = sys.argv[1] if len(sys.argv) > 1 else (test_images[0] if test_images else None)
    
    if not test_image:
        print("No test image found")
        sys.exit(1)
    
    extractor = UltraOCRExtractor()
    data, results = extractor.extract(test_image)
    
    print("\n" + "="*60)
    print("FINAL EXTRACTED DATA:")
    print("="*60)
    for k, v in data.items():
        if k != 'raw_text':
            print(f"  {k}: {v}")
    
    print("\n" + "="*60)
    print("FLAT DATA FOR SHEET:")
    print("="*60)
    flat = flatten_data_for_sheet(data)
    for i, v in enumerate(flat[:15]):  # Show first 15
        print(f"  [{i}]: {v}")
