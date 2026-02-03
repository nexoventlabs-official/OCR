"""Test extraction accuracy"""
from modules.groq_vision_ocr import GroqVisionOCR

e = GroqVisionOCR()
data, raw = e.extract('uploads/upload_20260203_000317.jpg')

print('='*60)
print('EXTRACTED vs EXPECTED:')
print('='*60)
print(f"Semester: {data.get('semester')} (Expected: V)")
print(f"Section: {data.get('section')} (Expected: B)")

pa = data.get('part_a_marks', {})
print(f"Part A Q1: {pa.get('q1')} (Expected: 1)")
print(f"Part A Q2: {pa.get('q2')} (Expected: 0)")
print(f"Part A Q3: {pa.get('q3')} (Expected: 1)")
print(f"Part A Q4: {pa.get('q4')} (Expected: 0)")
print(f"Part A Q5: {pa.get('q5')} (Expected: 1)")
print(f"Part A Total: {pa.get('total')} (Expected: 3)")

pb = data.get('part_bc_marks', {})
print(f"Part BC Q6: {pb.get('q6')} (Expected: 11)")
print(f"Part BC Q7: {pb.get('q7')} (Expected: 10)")
print(f"Part BC Q8: {pb.get('q8')} (Expected: 11)")
print(f"Part BC Q9: {pb.get('q9')} (Expected: 11)")
print(f"Part BC Total: {pb.get('total')} (Expected: 43)")

print(f"Grand Total: {data.get('grand_total')} (Expected: 46/60)")

co = data.get('course_outcomes', {})
print(f"CO4: {co.get('CO4', {}).get('total', '')} (Expected: 34)")
print(f"CO5: {co.get('CO5', {}).get('total', '')} (Expected: 12)")
