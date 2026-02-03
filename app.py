"""
Assessment OCR Scanner - Main Application
Flask web application for scanning assessment sheets and storing data in Google Sheets
"""
import os
import sys
import json
import base64
from datetime import datetime
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config, SHEET_HEADERS
from modules.sheets_manager import get_sheets_manager

# Check if running on server (no webcam available)
IS_SERVER = os.environ.get('RENDER', False) or os.environ.get('DOCKER', False)

# Import webcam module (may fail on server)
try:
    from modules.webcam_capture import get_webcam_manager, ImageCapture
    WEBCAM_AVAILABLE = not IS_SERVER  # Disable webcam on server
except ImportError:
    WEBCAM_AVAILABLE = False
    print("⚠ Webcam module not available (server environment)")

# Import Cloudinary storage
USE_CLOUDINARY = False
try:
    from modules.cloudinary_storage import get_cloudinary_storage, upload_to_cloudinary
    cloudinary_storage = get_cloudinary_storage()
    if cloudinary_storage.is_available():
        USE_CLOUDINARY = Config.USE_CLOUDINARY
        print(f"✓ Cloudinary storage {'enabled' if USE_CLOUDINARY else 'available but disabled'}")
    else:
        print("✗ Cloudinary not configured")
except ImportError as e:
    print(f"✗ Cloudinary module not available: {e}")
    cloudinary_storage = None

# Try Groq Vision first (most accurate), then fallback to other methods
USE_GROQ_VISION = False
USE_ULTRA_OCR = False
USE_ADVANCED_OCR = False

try:
    from modules.groq_vision_ocr import GroqVisionOCR, flatten_data_for_sheet, GROQ_AVAILABLE
    if GROQ_AVAILABLE:
        extractor = GroqVisionOCR()
        USE_GROQ_VISION = True
        print("✓ Using Groq Vision OCR (Most Accurate)")
    else:
        raise ImportError("Groq not available")
except ImportError as e:
    print(f"Groq Vision not available: {e}")
    try:
        from modules.ultra_ocr import UltraOCRExtractor, flatten_data_for_sheet
        extractor = UltraOCRExtractor()
        USE_ULTRA_OCR = True
        print("✓ Using Ultra Multi-Engine OCR")
    except ImportError as e:
        print(f"Ultra OCR not available: {e}")
        try:
            from modules.advanced_ocr import AssessmentFormParser, flatten_data_for_sheet
            extractor = AssessmentFormParser()
            USE_ADVANCED_OCR = True
            print("✓ Using Advanced Multi-Engine OCR")
        except ImportError as e:
            print(f"Advanced OCR not available: {e}")
            from modules.ocr_extractor import AssessmentDataExtractor, flatten_data_for_sheet
            extractor = AssessmentDataExtractor()
            print("✓ Using Basic OCR")

# Initialize Flask app
app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH

# Ensure upload folder exists
Config.ensure_upload_folder()

# Initialize modules
sheets_manager = get_sheets_manager()

# Initialize webcam manager only if available
webcam_manager = None
if WEBCAM_AVAILABLE:
    try:
        webcam_manager = get_webcam_manager()
        print("✓ Webcam manager initialized")
    except Exception as e:
        print(f"⚠ Webcam initialization failed: {e}")
        WEBCAM_AVAILABLE = False
else:
    print("⚠ Webcam disabled (server environment)")

# Stats tracking
app_stats = {
    'total_scans': 0,
    'successful_scans': 0,
    'today_scans': 0,
    'last_scan_date': None
}


def update_stats(success=True):
    """Update application statistics"""
    global app_stats
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    if app_stats['last_scan_date'] != today:
        app_stats['today_scans'] = 0
        app_stats['last_scan_date'] = today
    
    app_stats['total_scans'] += 1
    app_stats['today_scans'] += 1
    
    if success:
        app_stats['successful_scans'] += 1


# ==================== Web Routes ====================

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


# ==================== API Routes ====================

@app.route('/api/camera/start', methods=['POST'])
def start_camera():
    """Start the webcam"""
    if not WEBCAM_AVAILABLE or not webcam_manager:
        return jsonify({
            'success': False,
            'message': 'Webcam not available (server environment). Use image upload instead.'
        })
    success, message = webcam_manager.start()
    return jsonify({
        'success': success,
        'message': message
    })


@app.route('/api/camera/stop', methods=['POST'])
def stop_camera():
    """Stop the webcam"""
    if not WEBCAM_AVAILABLE or not webcam_manager:
        return jsonify({
            'success': False,
            'message': 'Webcam not available'
        })
    success, message = webcam_manager.stop()
    return jsonify({
        'success': success,
        'message': message
    })


@app.route('/api/camera/stream')
def camera_stream():
    """Stream webcam frames"""
    if not WEBCAM_AVAILABLE or not webcam_manager:
        return Response('Webcam not available on server', status=503)
    if not webcam_manager.is_active():
        return Response('Camera not active', status=503)
    
    return Response(
        webcam_manager.generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/camera/frame')
def get_frame():
    """Get single frame as JPEG"""
    if not WEBCAM_AVAILABLE or not webcam_manager:
        return jsonify({'success': False, 'message': 'Webcam not available on server'}), 503
    
    frame_bytes = webcam_manager.get_jpeg_frame()
    
    if frame_bytes:
        return Response(frame_bytes, mimetype='image/jpeg')
    
    return jsonify({'success': False, 'message': 'No frame available'}), 503


@app.route('/api/capture-and-extract', methods=['POST'])
def capture_and_extract():
    """Capture image from webcam and extract data"""
    if not WEBCAM_AVAILABLE or not webcam_manager:
        return jsonify({
            'success': False,
            'message': 'Webcam capture not available on server. Please use file upload instead.'
        }), 503
    
    try:
        # Capture image
        success, frame, message = webcam_manager.capture_image()
        
        if not success or frame is None:
            return jsonify({
                'success': False,
                'message': message or 'Failed to capture image'
            })
        
        # Save captured image (Cloudinary or local)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_url = None
        local_path = None
        
        import cv2
        
        if USE_CLOUDINARY and cloudinary_storage:
            # Upload to Cloudinary
            upload_success, image_url, upload_msg = cloudinary_storage.upload_frame(
                frame,
                folder="ocr_captures",
                prefix="capture",
                tags=['ocr', 'capture', 'webcam']
            )
            if not upload_success:
                print(f"Cloudinary upload failed: {upload_msg}, falling back to local storage")
                local_path = os.path.join(Config.UPLOAD_FOLDER, f'capture_{timestamp}.jpg')
                cv2.imwrite(local_path, frame)
        else:
            # Save locally
            local_path = os.path.join(Config.UPLOAD_FOLDER, f'capture_{timestamp}.jpg')
            cv2.imwrite(local_path, frame)
        
        # For OCR, we need a local file or use the frame directly
        # Save temporarily for OCR processing if using cloud storage
        temp_path = local_path
        if not temp_path:
            temp_path = os.path.join(Config.UPLOAD_FOLDER, f'temp_capture_{timestamp}.jpg')
            cv2.imwrite(temp_path, frame)
        
        # Extract data using appropriate method
        if USE_GROQ_VISION:
            data, raw_text = extractor.extract(temp_path)
        elif USE_ULTRA_OCR:
            data, ocr_results = extractor.extract(temp_path)
            raw_text = ocr_results.get('combined_text', '') if isinstance(ocr_results, dict) else str(ocr_results)
        elif USE_ADVANCED_OCR:
            data, ocr_results = extractor.parse_assessment(temp_path)
            raw_text = ocr_results.get('combined_text', '') if isinstance(ocr_results, dict) else str(ocr_results)
        else:
            data, raw_text = extractor.process_image(frame)
        
        # Clean up temp file if using cloud storage
        if USE_CLOUDINARY and image_url and temp_path and temp_path != local_path:
            try:
                os.remove(temp_path)
            except:
                pass
        
        if data:
            # Add image URL to data
            if image_url:
                data['image_url'] = image_url
            
            update_stats(True)
            return jsonify({
                'success': True,
                'data': data,
                'raw_text': raw_text,
                'image_path': image_url or local_path,
                'storage': 'cloudinary' if image_url else 'local'
            })
        else:
            update_stats(False)
            return jsonify({
                'success': False,
                'message': 'Failed to extract data from image'
            })
            
    except Exception as e:
        update_stats(False)
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        })


@app.route('/api/upload-and-extract', methods=['POST'])
def upload_and_extract():
    """Upload image and extract data"""
    try:
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'message': 'No image file provided'
            })
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({
                'success': False,
                'message': 'No file selected'
            })
        
        # Read image
        image_bytes = file.read()
        success, image, message = ImageCapture.image_from_bytes(image_bytes)
        
        if not success or image is None:
            return jsonify({
                'success': False,
                'message': message or 'Failed to load image'
            })
        
        # Save uploaded image (Cloudinary or local)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_url = None
        local_path = None
        
        import cv2
        
        if USE_CLOUDINARY and cloudinary_storage:
            # Upload to Cloudinary
            upload_success, image_url, upload_msg = cloudinary_storage.upload_frame(
                image,
                folder="ocr_uploads",
                prefix="upload",
                tags=['ocr', 'upload', 'document']
            )
            if not upload_success:
                print(f"Cloudinary upload failed: {upload_msg}, falling back to local storage")
                local_path = os.path.join(Config.UPLOAD_FOLDER, f'upload_{timestamp}.jpg')
                cv2.imwrite(local_path, image)
        else:
            # Save locally
            local_path = os.path.join(Config.UPLOAD_FOLDER, f'upload_{timestamp}.jpg')
            cv2.imwrite(local_path, image)
        
        # For OCR, we need a local file
        temp_path = local_path
        if not temp_path:
            temp_path = os.path.join(Config.UPLOAD_FOLDER, f'temp_upload_{timestamp}.jpg')
            cv2.imwrite(temp_path, image)
        
        # Extract data using appropriate method
        if USE_GROQ_VISION:
            data, raw_text = extractor.extract(temp_path)
        elif USE_ULTRA_OCR:
            data, ocr_results = extractor.extract(temp_path)
            raw_text = ocr_results.get('combined_text', '') if isinstance(ocr_results, dict) else str(ocr_results)
        elif USE_ADVANCED_OCR:
            data, ocr_results = extractor.parse_assessment(temp_path)
            raw_text = ocr_results.get('combined_text', '') if isinstance(ocr_results, dict) else str(ocr_results)
        else:
            data, raw_text = extractor.process_image(image)
        
        # Clean up temp file if using cloud storage
        if USE_CLOUDINARY and image_url and temp_path and temp_path != local_path:
            try:
                os.remove(temp_path)
            except:
                pass
        
        if data:
            # Add image URL to data
            if image_url:
                data['image_url'] = image_url
            
            update_stats(True)
            return jsonify({
                'success': True,
                'data': data,
                'raw_text': raw_text,
                'image_path': image_url or local_path,
                'storage': 'cloudinary' if image_url else 'local'
            })
        else:
            update_stats(False)
            return jsonify({
                'success': False,
                'message': 'Failed to extract data from image'
            })
            
    except Exception as e:
        update_stats(False)
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        })


@app.route('/api/save-to-sheet', methods=['POST'])
def save_to_sheet():
    """Save extracted data to Google Sheets"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'message': 'No data provided'
            })
        
        # Flatten data for sheet
        flat_data = flatten_data_for_sheet(data)
        
        # Add to sheet
        success, message = sheets_manager.add_record(flat_data)
        
        return jsonify({
            'success': success,
            'message': message
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error saving to sheet: {str(e)}'
        })


@app.route('/api/stats')
def get_stats():
    """Get application statistics"""
    try:
        record_count = sheets_manager.get_record_count()
        
        success_rate = 100
        if app_stats['total_scans'] > 0:
            success_rate = round(
                (app_stats['successful_scans'] / app_stats['total_scans']) * 100
            )
        
        return jsonify({
            'total_records': record_count,
            'today_scans': app_stats['today_scans'],
            'success_rate': success_rate,
            'total_scans': app_stats['total_scans']
        })
        
    except Exception as e:
        return jsonify({
            'total_records': 0,
            'today_scans': 0,
            'success_rate': 100,
            'error': str(e)
        })


@app.route('/api/recent-records')
def get_recent_records():
    """Get recent records from Google Sheets"""
    try:
        all_records = sheets_manager.get_all_records()
        
        # Get last 10 records
        recent = all_records[-10:] if len(all_records) > 10 else all_records
        recent.reverse()  # Most recent first
        
        # Format records
        formatted = []
        for record in recent:
            formatted.append({
                'timestamp': record.get('Timestamp', ''),
                'registration_no': record.get('Registration No', ''),
                'degree_branch': record.get('Degree/Branch', ''),
                'subject_code': record.get('Subject Code', ''),
                'total_marks': record.get('Total Marks', ''),
                'grand_total': record.get('Grand Total (out of 100)', '')
            })
        
        return jsonify({
            'success': True,
            'records': formatted
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'records': [],
            'error': str(e)
        })


@app.route('/api/sheet-url')
def get_sheet_url():
    """Get Google Sheet URL"""
    return jsonify({
        'url': sheets_manager.get_sheet_url()
    })


@app.route('/api/search', methods=['POST'])
def search_records():
    """Search records by registration number"""
    try:
        data = request.get_json()
        reg_no = data.get('registration_no', '')
        
        if not reg_no:
            return jsonify({
                'success': False,
                'message': 'Registration number required'
            })
        
        records = sheets_manager.search_by_registration(reg_no)
        
        return jsonify({
            'success': True,
            'records': records
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Search failed: {str(e)}'
        })


@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'camera_active': webcam_manager.is_active(),
        'sheets_connected': sheets_manager.check_connection(),
        'cloudinary_enabled': USE_CLOUDINARY,
        'cloudinary_available': cloudinary_storage.is_available() if cloudinary_storage else False,
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/storage/status')
def storage_status():
    """Get storage status (Cloudinary vs Local)"""
    return jsonify({
        'cloudinary_enabled': USE_CLOUDINARY,
        'cloudinary_available': cloudinary_storage.is_available() if cloudinary_storage else False,
        'local_folder': Config.UPLOAD_FOLDER,
        'storage_mode': 'cloudinary' if USE_CLOUDINARY and cloudinary_storage and cloudinary_storage.is_available() else 'local'
    })


@app.route('/api/storage/images')
def list_stored_images():
    """List images stored in Cloudinary"""
    if not USE_CLOUDINARY or not cloudinary_storage or not cloudinary_storage.is_available():
        return jsonify({
            'success': False,
            'message': 'Cloudinary storage not available',
            'images': []
        })
    
    try:
        folder = request.args.get('folder', 'ocr_captures')
        max_results = int(request.args.get('limit', 20))
        
        success, images, message = cloudinary_storage.list_images(folder, max_results)
        
        return jsonify({
            'success': success,
            'images': images,
            'message': message
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e),
            'images': []
        })


# ==================== Real-Time OCR Endpoints ====================

# Import real-time OCR module
try:
    from modules.realtime_ocr import (
        RealTimeOCRStream, 
        get_realtime_ocr, 
        stop_realtime_ocr
    )
    REALTIME_OCR_AVAILABLE = True
    print("✓ Real-Time OCR module loaded")
except ImportError as e:
    REALTIME_OCR_AVAILABLE = False
    print(f"✗ Real-Time OCR not available: {e}")

# Real-time OCR instance
realtime_ocr_instance = None


@app.route('/api/realtime-ocr/start', methods=['POST'])
def start_realtime_ocr():
    """Start real-time OCR stream."""
    global realtime_ocr_instance
    
    if not WEBCAM_AVAILABLE:
        return jsonify({
            'success': False,
            'message': 'Real-Time OCR requires webcam. Not available on server deployment.'
        })
    
    if not REALTIME_OCR_AVAILABLE:
        return jsonify({
            'success': False,
            'message': 'Real-Time OCR module not available'
        })
    
    try:
        data = request.get_json() or {}
        
        # Get configuration options
        camera_src = data.get('camera_src', 0)
        resolution = data.get('resolution', [1280, 720])
        crop = data.get('crop', [100, 100])
        view_mode = data.get('view_mode', 1)
        language = data.get('language', 'eng')
        use_easyocr = data.get('use_easyocr', False)
        
        # Stop existing instance if running
        if realtime_ocr_instance and realtime_ocr_instance.running:
            realtime_ocr_instance.stop()
        
        # Create new instance
        realtime_ocr_instance = RealTimeOCRStream(
            camera_src=camera_src,
            resolution=tuple(resolution),
            crop=tuple(crop) if crop else None,
            view_mode=view_mode,
            language=language,
            use_easyocr=use_easyocr
        )
        
        if realtime_ocr_instance.start():
            return jsonify({
                'success': True,
                'message': 'Real-Time OCR started',
                'config': {
                    'camera_src': camera_src,
                    'resolution': resolution,
                    'crop': crop,
                    'view_mode': view_mode,
                    'language': language
                }
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Failed to start Real-Time OCR'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        })


@app.route('/api/realtime-ocr/stop', methods=['POST'])
def stop_realtime_ocr_endpoint():
    """Stop real-time OCR stream."""
    global realtime_ocr_instance
    
    try:
        if realtime_ocr_instance:
            realtime_ocr_instance.stop()
            realtime_ocr_instance = None
        
        return jsonify({
            'success': True,
            'message': 'Real-Time OCR stopped'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        })


@app.route('/api/realtime-ocr/stream')
def realtime_ocr_stream():
    """Stream real-time OCR frames."""
    global realtime_ocr_instance
    
    if not realtime_ocr_instance or not realtime_ocr_instance.running:
        return Response('Real-Time OCR not active', status=503)
    
    return Response(
        realtime_ocr_instance.generate_frames_with_ocr(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


@app.route('/api/realtime-ocr/frame')
def realtime_ocr_frame():
    """Get single frame with OCR overlay."""
    global realtime_ocr_instance
    
    if not realtime_ocr_instance or not realtime_ocr_instance.running:
        return jsonify({'success': False, 'message': 'Real-Time OCR not active'}), 503
    
    frame_bytes = realtime_ocr_instance.get_jpeg_frame_with_ocr()
    
    if frame_bytes:
        return Response(frame_bytes, mimetype='image/jpeg')
    
    return jsonify({'success': False, 'message': 'No frame available'}), 503


@app.route('/api/realtime-ocr/text')
def realtime_ocr_text():
    """Get current detected text from real-time OCR."""
    global realtime_ocr_instance
    
    if not realtime_ocr_instance or not realtime_ocr_instance.running:
        return jsonify({
            'success': False,
            'text': '',
            'message': 'Real-Time OCR not active'
        })
    
    return jsonify({
        'success': True,
        'text': realtime_ocr_instance.detected_text,
        'captures': realtime_ocr_instance.captures
    })


@app.route('/api/realtime-ocr/capture', methods=['POST'])
def realtime_ocr_capture():
    """Capture current frame from real-time OCR stream."""
    global realtime_ocr_instance
    
    if not realtime_ocr_instance or not realtime_ocr_instance.running:
        return jsonify({
            'success': False,
            'message': 'Real-Time OCR not active'
        })
    
    try:
        success, frame, text = realtime_ocr_instance.get_frame_with_ocr()
        
        if not success or frame is None:
            return jsonify({
                'success': False,
                'message': 'Failed to capture frame'
            })
        
        # Save image (Cloudinary or local)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_url = None
        local_path = None
        
        import cv2
        
        if USE_CLOUDINARY and cloudinary_storage:
            # Upload to Cloudinary
            upload_success, image_url, upload_msg = cloudinary_storage.upload_frame(
                frame,
                folder="ocr_realtime",
                prefix="realtime",
                tags=['ocr', 'realtime', 'capture']
            )
            if not upload_success:
                print(f"Cloudinary upload failed: {upload_msg}, falling back to local storage")
                local_path = os.path.join(Config.UPLOAD_FOLDER, f'realtime_capture_{timestamp}.jpg')
                cv2.imwrite(local_path, frame)
        else:
            local_path = os.path.join(Config.UPLOAD_FOLDER, f'realtime_capture_{timestamp}.jpg')
            cv2.imwrite(local_path, frame)
        
        return jsonify({
            'success': True,
            'image_path': image_url or local_path,
            'detected_text': text,
            'storage': 'cloudinary' if image_url else 'local',
            'message': 'Frame captured successfully'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        })


@app.route('/api/realtime-ocr/view-mode', methods=['POST'])
def set_realtime_ocr_view_mode():
    """Change real-time OCR view mode."""
    global realtime_ocr_instance
    
    if not realtime_ocr_instance:
        return jsonify({
            'success': False,
            'message': 'Real-Time OCR not active'
        })
    
    try:
        data = request.get_json()
        mode = data.get('mode', 1)
        
        if mode in [1, 2, 3, 4]:
            realtime_ocr_instance.view_mode = mode
            return jsonify({
                'success': True,
                'view_mode': mode,
                'message': f'View mode changed to {mode}'
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Invalid view mode. Use 1-4.'
            })
            
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error: {str(e)}'
        })


@app.route('/api/realtime-ocr/status')
def realtime_ocr_status():
    """Get real-time OCR status."""
    global realtime_ocr_instance
    
    if realtime_ocr_instance and realtime_ocr_instance.running:
        return jsonify({
            'active': True,
            'view_mode': realtime_ocr_instance.view_mode,
            'language': realtime_ocr_instance.language,
            'captures': realtime_ocr_instance.captures,
            'detected_text': realtime_ocr_instance.detected_text[:200]  # Limit length
        })
    else:
        return jsonify({
            'active': False
        })


# ==================== Error Handlers ====================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404


@app.errorhandler(500)
def server_error(error):
    return jsonify({'error': 'Internal server error'}), 500


# ==================== Main ====================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("  Assessment OCR Scanner")
    print("  Scanning documents and saving to Google Sheets")
    print("="*60)
    
    # Check connections
    print("\n[*] Checking connections...")
    
    if sheets_manager.check_connection():
        print("[✓] Google Sheets connected")
        print(f"    Sheet URL: {sheets_manager.get_sheet_url()}")
    else:
        print("[✗] Google Sheets connection failed")
        print("    Please check your credentials in .env file")
    
    print("\n[*] Starting web server...")
    print("[*] Open http://localhost:5000 in your browser")
    print("[*] Press Ctrl+C to stop\n")
    
    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
