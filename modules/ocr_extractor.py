"""
OCR Extraction Module
Extracts text and structured data from assessment sheet images
Uses Groq Vision API for enhanced AI-powered extraction
"""
import re
import cv2
import numpy as np
from PIL import Image
import pytesseract
from datetime import datetime
import os
import base64
import json

from config import Config

# Set Tesseract path
if os.path.exists(Config.TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = Config.TESSERACT_PATH

# Import Groq
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("Groq not installed. Install with: pip install groq")


class GroqVisionExtractor:
    """Use Groq Vision API for AI-powered data extraction"""
    
    def __init__(self):
        self.client = None
        if GROQ_AVAILABLE and Config.GROQ_API_KEY:
            self.client = Groq(api_key=Config.GROQ_API_KEY)
    
    def encode_image_to_base64(self, image):
        """Convert OpenCV image to base64 string"""
        if isinstance(image, str):
            # It's a file path
            with open(image, "rb") as f:
                return base64.b64encode(f.read()).decode('utf-8')
        else:
            # It's an OpenCV image array
            _, buffer = cv2.imencode('.jpg', image)
            return base64.b64encode(buffer).decode('utf-8')
    
    def extract_with_vision(self, image):
        """Extract assessment data using Groq Vision API"""
        if not self.client:
            return None, "Groq client not initialized. Check API key."
        
        try:
            # Encode image
            base64_image = self.encode_image_to_base64(image)
            
            # Create the prompt for structured extraction
            extraction_prompt = """Analyze this assessment/exam sheet image and extract ALL information into a structured JSON format.

Look for and extract these fields (use empty string if not found):
- registration_no: Student registration number (usually 12+ digits)
- college_code: 3-4 digit college code
- date: Date on the form
- degree_branch: Degree and branch (e.g., BE/ECE, B.Tech/CSE)
- semester_section: Semester and section (e.g., VI 'B')
- subject_code: Subject code (e.g., CEC367)
- subject_name: Subject name (e.g., Industrial IoT and Industry 4.0)
- no_of_pages: Number of pages used
- hall_superintendent: Name of hall superintendent

For marks, extract:
- part_a_marks: Object with q1, q2, q3, q4, q5 marks (Part A questions, usually 1-2 marks each)
- part_bc_marks: Object with marks for questions 6-9 (Part B & C), each may have sub-parts a/b with marks I, II, III
- total_marks: Total marks obtained
- grand_total: Grand total (often shown as X/100 or circled)

Course outcomes:
- course_outcomes: Object with CO1, CO2, etc. Each CO has part_a, part_b, part_c, total

Other:
- examiner_name: Name of the examiner

IMPORTANT: 
- Look carefully at handwritten numbers in RED ink - these are typically the marks
- The grand total is often circled (like 46/60)
- Part A is usually at the top, Part B&C below
- Return ONLY valid JSON, no markdown or explanation

Example output format:
{
    "registration_no": "113383106069",
    "college_code": "1133",
    "date": "09/10/2025",
    "degree_branch": "BE/ECE",
    "semester_section": "VI 'B'",
    "subject_code": "CEC367",
    "subject_name": "INDUSTRIAL IOT AND INDUSTRY 4.0",
    "no_of_pages": "02",
    "hall_superintendent": "",
    "part_a_marks": {"q1": "1", "q2": "0", "q3": "0", "q4": "0", "q5": "1"},
    "part_bc_marks": {},
    "total_marks": "43",
    "grand_total": "46/60",
    "course_outcomes": {
        "CO1": {"part_a": "2", "part_b": "21", "part_c": "11", "total": "34"},
        "CO2": {"part_a": "1", "part_b": "1", "part_c": "", "total": "12"}
    },
    "examiner_name": ""
}"""

            # Call Groq Vision API
            response = self.client.chat.completions.create(
                model="llama-3.2-90b-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": extraction_prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=2000,
                temperature=0.1
            )
            
            # Parse response
            response_text = response.choices[0].message.content
            
            # Try to extract JSON from response
            try:
                # Clean up response - remove markdown code blocks if present
                json_text = response_text
                if "```json" in json_text:
                    json_text = json_text.split("```json")[1].split("```")[0]
                elif "```" in json_text:
                    json_text = json_text.split("```")[1].split("```")[0]
                
                data = json.loads(json_text.strip())
                data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                data['raw_text'] = response_text
                return data, response_text
                
            except json.JSONDecodeError as e:
                print(f"JSON parse error: {e}")
                print(f"Response was: {response_text}")
                return None, response_text
                
        except Exception as e:
            error_msg = f"Groq Vision API error: {str(e)}"
            print(error_msg)
            return None, error_msg


class ImagePreprocessor:
    """Preprocess images for better OCR accuracy"""
    
    @staticmethod
    def preprocess_image(image):
        """Apply preprocessing to improve OCR accuracy"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply denoising
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Apply adaptive thresholding
        thresh = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Deskew if needed
        coords = np.column_stack(np.where(thresh > 0))
        if len(coords) > 0:
            angle = cv2.minAreaRect(coords)[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle
            
            if abs(angle) > 0.5 and abs(angle) < 10:
                (h, w) = thresh.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                thresh = cv2.warpAffine(thresh, M, (w, h),
                    flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        
        return thresh
    
    @staticmethod
    def enhance_for_handwriting(image):
        """Enhanced preprocessing for handwritten text"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Increase contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Sharpen
        kernel = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # Binary threshold
        _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary


class AssessmentDataExtractor:
    """Extract structured data from assessment sheet OCR text"""
    
    def __init__(self):
        self.preprocessor = ImagePreprocessor()
        self.groq_extractor = GroqVisionExtractor()
        self.use_groq = GROQ_AVAILABLE and Config.GROQ_API_KEY
    
    def extract_text_from_image(self, image):
        """Extract raw text from image using OCR"""
        # Preprocess image
        processed = self.preprocessor.preprocess_image(image)
        
        # Configure Tesseract for better accuracy
        custom_config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
        
        # Extract text
        text = pytesseract.image_to_string(processed, config=custom_config)
        
        # Also try with handwriting enhancement
        enhanced = self.preprocessor.enhance_for_handwriting(image)
        text_enhanced = pytesseract.image_to_string(enhanced, config=custom_config)
        
        # Combine results
        return text + "\n" + text_enhanced
    
    def extract_structured_data(self, raw_text, image=None):
        """Parse OCR text and extract structured assessment data"""
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
            'raw_text': raw_text
        }
        
        # Clean text for processing
        text = raw_text.upper()
        lines = [line.strip() for line in raw_text.split('\n') if line.strip()]
        
        # Extract Registration Number
        reg_patterns = [
            r'REGISTRATION\s*(?:NO\.?|NUMBER)?\s*[:\s]*(\d{12,})',
            r'REG(?:\.?\s*)?(?:NO\.?)?\s*[:\s]*(\d{12,})',
            r'(\d{12}\d*)'
        ]
        for pattern in reg_patterns:
            match = re.search(pattern, text)
            if match:
                data['registration_no'] = match.group(1)
                break
        
        # Extract College Code
        college_patterns = [
            r'COLLEGE\s*CODE\s*[:\s\|]*(\d{3,4})',
            r'CODE\s*[:\s\|]*(\d{4})',
        ]
        for pattern in college_patterns:
            match = re.search(pattern, text)
            if match:
                data['college_code'] = match.group(1)
                break
        
        # Extract Date
        date_patterns = [
            r'DATE\s*[:\s]*(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
            r'(\d{1,2}[/\-\.]\d{1,2}[/\-\.]\d{2,4})',
        ]
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                data['date'] = match.group(1)
                break
        
        # Extract Degree/Branch
        degree_patterns = [
            r'DEGREE\s*/?\s*BRANCH\s*[:\s]*([A-Z]+\s*/?\s*[A-Z]+)',
            r'(BE\s*/?\s*ECE|BE\s*/?\s*CSE|BE\s*/?\s*EEE|BE\s*/?\s*MECH|B\.?TECH)',
        ]
        for pattern in degree_patterns:
            match = re.search(pattern, text)
            if match:
                data['degree_branch'] = match.group(1).strip()
                break
        
        # Extract Semester/Section
        semester_patterns = [
            r'SEMESTER\s*/?\s*SECTION\s*[:\s]*([IVXLC]+\s*[\'"]?\s*[A-Z]?)',
            r'SEM(?:ESTER)?\s*[:\s]*([IVXLC]+)',
            r'SECTION\s*[:\s]*([A-Z])',
        ]
        for pattern in semester_patterns:
            match = re.search(pattern, text)
            if match:
                data['semester_section'] = match.group(1).strip()
                break
        
        # Extract Subject Code and Name
        subject_patterns = [
            r'SUBJECT\s*CODE\s*/?\s*NAME\s*[:\s]*([A-Z]{2,3}\d{3})\s*/?\s*(.+?)(?:NO\.?\s*OF|$)',
            r'([A-Z]{2,3}\d{3})\s*/?\s*(INDUSTRIAL\s*IOT|[A-Z\s]+(?:4\.0)?)',
        ]
        for pattern in subject_patterns:
            match = re.search(pattern, text)
            if match:
                data['subject_code'] = match.group(1).strip()
                if match.lastindex >= 2:
                    data['subject_name'] = match.group(2).strip()
                break
        
        # Extract Number of Pages
        pages_patterns = [
            r'NO\.?\s*OF\s*PAGES\s*(?:USED)?\s*[:\s]*(\d+)',
            r'PAGES\s*[:\s]*(\d+)',
        ]
        for pattern in pages_patterns:
            match = re.search(pattern, text)
            if match:
                data['no_of_pages'] = match.group(1)
                break
        
        # Extract Part A marks (Questions 1-5)
        part_a_marks = {}
        for i in range(1, 6):
            mark_patterns = [
                rf'(?:Q\.?|QUESTION)?\s*{i}\s*[:\s]*(\d+)',
                rf'PART\s*A.*?{i}\s*[:\s]*(\d+)',
            ]
            for pattern in mark_patterns:
                match = re.search(pattern, text)
                if match:
                    part_a_marks[f'q{i}'] = match.group(1)
                    break
        data['part_a_marks'] = part_a_marks
        
        # Extract marks from the document using number patterns
        numbers_found = re.findall(r'\b(\d{1,2})\b', raw_text)
        
        # Extract Total Marks
        total_patterns = [
            r'TOTAL\s*[:\s]*(\d+)',
            r'GRAND\s*TOTAL\s*[:\s]*(\d+)',
        ]
        for pattern in total_patterns:
            match = re.search(pattern, text)
            if match:
                data['total_marks'] = match.group(1)
                break
        
        # Extract Grand Total (circled value like 46/60)
        grand_total_patterns = [
            r'(\d+)\s*/\s*(\d+)',
            r'(\d+)\s*OUT\s*OF\s*(\d+)',
        ]
        for pattern in grand_total_patterns:
            match = re.search(pattern, text)
            if match:
                data['grand_total'] = f"{match.group(1)}/{match.group(2)}"
                break
        
        # Extract Course Outcomes
        co_patterns = [
            r'CO[-\s]*(\d)\s*[:\s]*(\d+)\s*[,\s]*(\d+)\s*[,\s]*(\d+)\s*[,\s]*(\d+)',
        ]
        for pattern in co_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                co_num = match[0]
                data['course_outcomes'][f'CO{co_num}'] = {
                    'part_a': match[1],
                    'part_b': match[2],
                    'part_c': match[3],
                    'total': match[4]
                }
        
        # Extract Examiner Name
        examiner_patterns = [
            r'NAME\s*OF\s*(?:THE\s*)?EXAMINER\s*[:\s]*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'EXAMINER\s*[:\s]*([A-Z][a-z]+)',
        ]
        for pattern in examiner_patterns:
            match = re.search(pattern, raw_text, re.IGNORECASE)
            if match:
                data['examiner_name'] = match.group(1).strip()
                break
        
        return data
    
    def process_image(self, image_path_or_array):
        """Main method to process an image and extract all data"""
        # Load image if path is provided
        if isinstance(image_path_or_array, str):
            image = cv2.imread(image_path_or_array)
            image_path = image_path_or_array
        else:
            image = image_path_or_array
            image_path = None
        
        if image is None:
            return None, "Failed to load image"
        
        # Try Groq Vision first if available
        if self.use_groq:
            print("Using Groq Vision API for extraction...")
            data, raw_text = self.groq_extractor.extract_with_vision(
                image_path if image_path else image
            )
            if data:
                print("✓ Groq Vision extraction successful")
                return data, raw_text
            else:
                print("✗ Groq Vision failed, falling back to Tesseract OCR")
        
        # Fallback to traditional OCR
        print("Using Tesseract OCR for extraction...")
        raw_text = self.extract_text_from_image(image)
        structured_data = self.extract_structured_data(raw_text, image)
        
        return structured_data, raw_text


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
    flat_data.append('')
    
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


# Test the extractor
if __name__ == "__main__":
    import sys
    
    # Check for uploads folder images
    uploads_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'uploads')
    
    # Find test image
    test_images = []
    if os.path.exists(uploads_folder):
        for f in os.listdir(uploads_folder):
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                test_images.append(os.path.join(uploads_folder, f))
    
    if len(sys.argv) > 1:
        test_image = sys.argv[1]
    elif test_images:
        test_image = test_images[0]
        print(f"Found image in uploads: {test_image}")
    else:
        test_image = "test_image.jpg"
    
    print(f"\n{'='*60}")
    print(f"Testing OCR Extractor")
    print(f"{'='*60}")
    print(f"Image: {test_image}")
    print(f"Groq API Available: {GROQ_AVAILABLE}")
    print(f"Groq API Key Set: {bool(Config.GROQ_API_KEY)}")
    print(f"{'='*60}\n")
    
    if os.path.exists(test_image):
        extractor = AssessmentDataExtractor()
        data, raw_text = extractor.process_image(test_image)
        
        if data:
            print("\n" + "="*60)
            print("EXTRACTED DATA:")
            print("="*60)
            for key, value in data.items():
                if key != 'raw_text':
                    print(f"  {key}: {value}")
            
            print("\n" + "="*60)
            print("RAW RESPONSE (first 1000 chars):")
            print("="*60)
            print(raw_text[:1000] if raw_text else "No raw text")
        else:
            print("Failed to extract data")
            print(f"Error: {raw_text}")
    else:
        print(f"Image not found: {test_image}")
        print(f"Available images in uploads: {test_images}")
