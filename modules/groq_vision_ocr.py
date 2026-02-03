"""
Groq Vision OCR Module - Uses Groq's Vision API for accurate assessment extraction
"""
import os
import base64
import json
import re
from datetime import datetime

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv('GROQ_API_KEY', '')

# Check if Groq is available
try:
    from groq import Groq
    GROQ_AVAILABLE = bool(GROQ_API_KEY and len(GROQ_API_KEY) > 10)
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠ Groq package not installed. Run: pip install groq")


class GroqVisionOCR:
    """Extract assessment data using Groq's Vision API"""
    
    # Detailed prompt for accurate extraction
    EXTRACTION_PROMPT = """You are an expert OCR system for Indian university internal assessment answer sheets.

TASK: Extract ALL data from this assessment sheet with 100% accuracy.

IMPORTANT READING INSTRUCTIONS:
1. The form has PRINTED labels and HANDWRITTEN values - read the HANDWRITTEN values carefully
2. Handwritten numbers may be in RED or BLUE ink
3. Look for marks written inside boxes, circles, or table cells
4. Roman numerals for semester: I=1, II=2, III=3, IV=4, V=5, VI=6, VII=7, VIII=8
5. Part A questions are worth 2 marks EACH (5 questions × 2 = 10 marks total)
6. Part B&C questions are worth 10-15 marks each (usually 4 questions totaling ~50 marks)

FORM STRUCTURE TO LOOK FOR:
- Top section: Registration No, College Code, Date, Degree/Branch, Semester, Section
- Middle: Subject Code, Subject Name, Assessment Type (I/II/III), Pages
- PART A table: 5 questions (Q1-Q5), each worth 0, 1, or 2 marks
- PART B & C table: Questions Q6-Q9, each worth ~10-15 marks
- Bottom: Course Outcomes (CO) table with CO1-CO5 totals, Examiner signature

Extract and return this EXACT JSON structure:

{
    "institution": "Full name of institution",
    "college_code": "4-digit code like 1133",
    "registration_no": "10-12 digit student registration number",
    "date": "DD/MM/YYYY format",
    "degree_branch": "e.g., B.E/ECE",
    "semester": "Roman numeral (I, II, III, IV, V, VI, VII, VIII)",
    "section": "Letter (A, B, C, D, E)",
    "subject_code": "e.g., CEC367",
    "subject_name": "Full subject name",
    "assessment_type": "I, II, or III",
    "no_of_pages": "Number",
    "part_a_marks": {
        "q1": "0, 1, or 2",
        "q2": "0, 1, or 2",
        "q3": "0, 1, or 2",
        "q4": "0, 1, or 2",
        "q5": "0, 1, or 2",
        "total": "Sum of q1-q5 (max 10)"
    },
    "part_bc_marks": {
        "q6": "marks (typically 10-15)",
        "q7": "marks (typically 10-15)",
        "q8": "marks (typically 10-15)",
        "q9": "marks (typically 10-15)",
        "total": "Sum of q6-q9 (max ~50)"
    },
    "total_marks": "Part A + Part B&C total",
    "max_marks": "Maximum possible (usually 60)",
    "grand_total": "Format: obtained/max (e.g., 46/60)",
    "course_outcomes": {
        "CO1": {"total": "marks or empty"},
        "CO2": {"total": "marks or empty"},
        "CO3": {"total": "marks or empty"},
        "CO4": {"total": "marks - look carefully, could be 30+"},
        "CO5": {"total": "marks - look carefully, could be 10+"}
    },
    "examiner_name": "Name if visible",
    "hall_superintendent": "Name if visible"
}

VERIFICATION RULES:
- Part A total should equal sum of Q1+Q2+Q3+Q4+Q5
- Part B&C total should equal sum of Q6+Q7+Q8+Q9
- Grand total = Part A total + Part B&C total
- CO totals should add up close to the grand total

Return ONLY valid JSON, no other text. Read each number very carefully!"""

    def __init__(self):
        self.client = None
        if GROQ_AVAILABLE:
            self.client = Groq(api_key=GROQ_API_KEY)
            print("✓ Groq Vision OCR initialized")
        else:
            print("✗ Groq Vision OCR not available")
    
    def encode_image(self, image_path):
        """Convert image to base64"""
        with open(image_path, "rb") as image_file:
            return base64.standard_b64encode(image_file.read()).decode("utf-8")
    
    def extract(self, image_path):
        """Extract data from image using Groq Vision"""
        if not self.client:
            return None, "Groq client not initialized"
        
        try:
            # Encode image
            base64_image = self.encode_image(image_path)
            
            # Determine image type
            ext = os.path.splitext(image_path)[1].lower()
            media_type = "image/jpeg" if ext in ['.jpg', '.jpeg'] else "image/png"
            
            print(f"  Sending image to Groq Vision API...")
            
            # Call Groq Vision API
            response = self.client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": self.EXTRACTION_PROMPT
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{media_type};base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                temperature=0.0,
                max_tokens=2500
            )
            
            # Get response text
            result_text = response.choices[0].message.content
            print(f"  Groq Vision response received")
            
            # Parse JSON from response
            data = self.parse_json_response(result_text)
            
            if data:
                # Add timestamp
                data['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                data['extraction_method'] = 'groq_vision'
                
                # Validate and correct Part A marks
                data = self.validate_marks(data)
                
                return data, result_text
            else:
                return None, f"Failed to parse response: {result_text}"
                
        except Exception as e:
            error_msg = str(e)
            print(f"  Groq Vision error: {error_msg}")
            return None, error_msg
    
    def validate_marks(self, data):
        """Validate and correct extracted marks"""
        # Part A validation
        part_a = data.get('part_a_marks', {})
        if part_a:
            total_str = str(part_a.get('total', '0'))
            total = int(total_str) if total_str.isdigit() else 0
            
            # Calculate sum of individual questions
            q_sum = 0
            for i in range(1, 6):
                q_val = str(part_a.get(f'q{i}', '0'))
                q_sum += int(q_val) if q_val.isdigit() else 0
            
            # If sum doesn't match total, log warning
            if q_sum != total:
                print(f"  Warning: Part A sum ({q_sum}) != total ({total})")
        
        # Part B&C validation
        part_bc = data.get('part_bc_marks', {})
        if part_bc:
            total_str = str(part_bc.get('total', '0'))
            total = int(total_str) if total_str.isdigit() else 0
            
            q_sum = 0
            for i in range(6, 10):
                q_val = str(part_bc.get(f'q{i}', '0'))
                q_sum += int(q_val) if q_val.isdigit() else 0
            
            if q_sum != total:
                print(f"  Warning: Part B&C sum ({q_sum}) != total ({total})")
        
        # Grand total validation
        part_a_total = int(str(part_a.get('total', '0'))) if str(part_a.get('total', '0')).isdigit() else 0
        part_bc_total = int(str(part_bc.get('total', '0'))) if str(part_bc.get('total', '0')).isdigit() else 0
        
        grand_total_str = str(data.get('grand_total', ''))
        if '/' in grand_total_str:
            obtained = grand_total_str.split('/')[0]
            if obtained.isdigit():
                calculated = part_a_total + part_bc_total
                if calculated != int(obtained):
                    print(f"  Warning: Calculated total ({calculated}) != Grand total ({obtained})")
        
        return data
    
    def parse_json_response(self, text):
        """Extract JSON from response text"""
        try:
            # Try direct JSON parse
            return json.loads(text)
        except:
            pass
        
        # Try to find JSON in text
        try:
            # Look for JSON block
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # Try to extract between code blocks
        try:
            code_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
            if code_match:
                return json.loads(code_match.group(1))
        except:
            pass
        
        return None


def flatten_data_for_sheet(data):
    """Convert extracted data to flat list for Google Sheets"""
    if not data:
        return []
    
    flat_data = [
        data.get('timestamp', ''),
        data.get('registration_no', ''),
        data.get('college_code', ''),
        data.get('date', ''),
        data.get('degree_branch', ''),
        f"{data.get('semester', '')}/{data.get('section', '')}",
        data.get('subject_code', ''),
        data.get('subject_name', ''),
        data.get('no_of_pages', ''),
        data.get('hall_superintendent', ''),
    ]
    
    # Part A marks (Q1-Q5)
    part_a = data.get('part_a_marks', {})
    if isinstance(part_a, dict):
        for i in range(1, 6):
            flat_data.append(str(part_a.get(f'q{i}', '')))
        flat_data.append(str(part_a.get('total', '')))
    else:
        flat_data.extend([''] * 6)
    
    # Part B&C marks (Q6-Q9)
    part_bc = data.get('part_bc_marks', {})
    if isinstance(part_bc, dict):
        for i in range(6, 10):
            q_marks = str(part_bc.get(f'q{i}', ''))
            # Add 3 sub-columns per question (I, II, III)
            flat_data.extend([q_marks, '', ''])
        flat_data.append(str(part_bc.get('total', '')))
    else:
        flat_data.extend([''] * 13)
    
    # Total Marks
    flat_data.append(str(data.get('total_marks', '')))
    
    # Grand Total
    flat_data.append(str(data.get('grand_total', '')))
    
    # Course Outcomes
    course_outcomes = data.get('course_outcomes', {})
    if isinstance(course_outcomes, dict):
        for co in ['CO1', 'CO2', 'CO3', 'CO4', 'CO5']:
            co_data = course_outcomes.get(co, {})
            if isinstance(co_data, dict):
                flat_data.extend([
                    str(co_data.get('part_a', '')),
                    str(co_data.get('part_b', '')),
                    str(co_data.get('part_c', '')),
                    str(co_data.get('total', ''))
                ])
            elif isinstance(co_data, str):
                flat_data.extend(['', '', '', co_data])
            else:
                flat_data.extend([''] * 4)
    else:
        flat_data.extend([''] * 20)
    
    # Examiner Name
    flat_data.append(str(data.get('examiner_name', '')))
    
    # Raw text / extraction method
    flat_data.append(data.get('extraction_method', 'groq_vision'))
    
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
    
    print("\n" + "="*60)
    print("GROQ VISION OCR TEST")
    print("="*60)
    print(f"Image: {test_image}")
    print("-"*60)
    
    extractor = GroqVisionOCR()
    data, raw_response = extractor.extract(test_image)
    
    if data:
        print("\n" + "="*60)
        print("EXTRACTED DATA:")
        print("="*60)
        for k, v in data.items():
            if k not in ['extraction_method']:
                print(f"  {k}: {v}")
        
        print("\n" + "="*60)
        print("FLAT DATA FOR SHEET:")
        print("="*60)
        flat = flatten_data_for_sheet(data)
        for i, v in enumerate(flat[:20]):
            print(f"  [{i}]: {v}")
    else:
        print(f"\nExtraction failed: {raw_response}")
