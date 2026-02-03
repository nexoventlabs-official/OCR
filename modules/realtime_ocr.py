"""
Production-Level Real-Time OCR Module
Implements multi-threaded video capture and OCR processing for seamless real-time text detection.

Based on the architecture from nathanaday/RealTime-OCR with production enhancements:
- Threaded video capture for smooth frame display
- Threaded OCR processing in background
- Multiple view modes for text detection visualization
- Frame rate monitoring
- Confidence-based text filtering
- Red ink extraction for handwritten marks
- Graceful shutdown handling
"""
import os
import sys
import time
import threading
from threading import Thread, Event
from datetime import datetime
from pathlib import Path
from queue import Queue
from typing import Optional, Tuple, List, Dict, Any

import cv2
import numpy as np

# OCR Engines
PYTESSERACT_AVAILABLE = False
EASYOCR_AVAILABLE = False

try:
    import pytesseract
    # Set Tesseract path for Windows
    tesseract_path = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    if os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
    PYTESSERACT_AVAILABLE = True
except ImportError:
    print("pytesseract not installed. Install with: pip install pytesseract")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    pass


class RateCounter:
    """
    Thread-safe counter for measuring iterations per second.
    Useful for monitoring frame rate and OCR processing speed.
    """
    
    def __init__(self):
        self.start_time: Optional[float] = None
        self.iterations: int = 0
        self._lock = threading.Lock()
    
    def start(self) -> 'RateCounter':
        """Start the rate counter."""
        with self._lock:
            self.start_time = time.perf_counter()
            self.iterations = 0
        return self
    
    def increment(self) -> None:
        """Increment the iteration count."""
        with self._lock:
            self.iterations += 1
    
    def rate(self) -> float:
        """Calculate current iterations per second."""
        with self._lock:
            if self.start_time is None or self.iterations == 0:
                return 0.0
            elapsed = time.perf_counter() - self.start_time
            return self.iterations / elapsed if elapsed > 0 else 0.0
    
    def reset(self) -> None:
        """Reset the counter."""
        with self._lock:
            self.start_time = time.perf_counter()
            self.iterations = 0


class ThreadedVideoStream:
    """
    Threaded video capture class for real-time frame acquisition.
    
    Runs frame capture in a dedicated thread to prevent blocking
    the main display loop, ensuring smooth video playback even
    during OCR processing.
    
    Attributes:
        src: Video source (camera index or video file path)
        frame: Most recently captured frame
        stopped: Flag indicating if capture should stop
    """
    
    def __init__(self, src: int = 0, resolution: Tuple[int, int] = (1280, 720)):
        """
        Initialize the video stream.
        
        Args:
            src: Camera index (0 for default webcam) or video file path
            resolution: Desired capture resolution (width, height)
        """
        self.src = src
        self.resolution = resolution
        self.stream: Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray] = None
        self.grabbed: bool = False
        self.stopped: bool = False
        self._lock = threading.Lock()
        self._stop_event = Event()
        self._thread: Optional[Thread] = None
    
    def start(self) -> 'ThreadedVideoStream':
        """Start the video capture thread."""
        # Initialize video capture
        self.stream = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)  # DirectShow for Windows
        
        if not self.stream.isOpened():
            # Fallback without DirectShow
            self.stream = cv2.VideoCapture(self.src)
        
        if not self.stream.isOpened():
            raise RuntimeError(f"Failed to open video source: {self.src}")
        
        # Set resolution
        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        # Set buffer size to reduce latency
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Read first frame
        self.grabbed, self.frame = self.stream.read()
        self.stopped = False
        self._stop_event.clear()
        
        # Start capture thread
        self._thread = Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        
        return self
    
    def _capture_loop(self) -> None:
        """Continuous frame capture loop running in separate thread."""
        while not self._stop_event.is_set():
            if self.stream is None or not self.stream.isOpened():
                break
                
            grabbed, frame = self.stream.read()
            
            if grabbed:
                with self._lock:
                    self.grabbed = grabbed
                    self.frame = frame
            
            # Small delay to prevent CPU hogging
            time.sleep(0.001)
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Get the most recent frame."""
        with self._lock:
            if self.frame is not None:
                return self.grabbed, self.frame.copy()
            return False, None
    
    def get_dimensions(self) -> Tuple[int, int]:
        """Get video dimensions (width, height)."""
        if self.stream:
            width = int(self.stream.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return width, height
        return 0, 0
    
    def stop(self) -> None:
        """Stop the video capture thread."""
        self._stop_event.set()
        self.stopped = True
        
        if self._thread:
            self._thread.join(timeout=2.0)
        
        if self.stream:
            self.stream.release()
            self.stream = None
    
    def is_active(self) -> bool:
        """Check if video stream is active."""
        return not self.stopped and self.stream is not None and self.stream.isOpened()


class ThreadedOCR:
    """
    Threaded OCR processor for real-time text detection.
    
    Runs OCR processing in a dedicated thread, continuously
    grabbing frames from the video stream and performing
    text detection without blocking the display.
    
    Attributes:
        boxes: Latest OCR detection results
        text: Detected text from latest frame
        stopped: Flag indicating if processing should stop
    """
    
    def __init__(self, use_easyocr: bool = False, language: str = 'eng'):
        """
        Initialize OCR processor.
        
        Args:
            use_easyocr: Whether to use EasyOCR instead of Tesseract
            language: Language code for OCR (e.g., 'eng', 'chi_sim+eng')
        """
        self.use_easyocr = use_easyocr and EASYOCR_AVAILABLE
        self.language = language
        
        # OCR state
        self.boxes: Optional[str] = None
        self.text: str = ""
        self.confidence: float = 0.0
        self.stopped: bool = False
        
        # Thread synchronization
        self._lock = threading.Lock()
        self._stop_event = Event()
        self._thread: Optional[Thread] = None
        
        # Video stream reference
        self.exchange: Optional[ThreadedVideoStream] = None
        
        # Processing region
        self.crop_region: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h
        
        # EasyOCR reader
        self._easyocr_reader = None
        if self.use_easyocr:
            print("Initializing EasyOCR reader...")
            self._easyocr_reader = easyocr.Reader(['en'], gpu=False)
            print("EasyOCR ready")
    
    def set_video_stream(self, video_stream: ThreadedVideoStream) -> None:
        """Set the video stream to grab frames from."""
        self.exchange = video_stream
    
    def set_crop_region(self, x: int, y: int, width: int, height: int) -> None:
        """Set region of interest for OCR (improves speed)."""
        self.crop_region = (x, y, width, height)
    
    def set_language(self, language: str) -> None:
        """Set OCR language."""
        self.language = language
    
    def start(self) -> 'ThreadedOCR':
        """Start the OCR processing thread."""
        self.stopped = False
        self._stop_event.clear()
        
        self._thread = Thread(target=self._ocr_loop, daemon=True)
        self._thread.start()
        
        return self
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame for better OCR accuracy.
        
        Args:
            frame: BGR frame from video capture
            
        Returns:
            Preprocessed grayscale frame
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply adaptive thresholding for better text detection
        # Commented by default as it may not work for all scenarios
        # gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                               cv2.THRESH_BINARY, 11, 2)
        
        return gray
    
    def _extract_red_ink(self, frame: np.ndarray) -> np.ndarray:
        """Extract red handwritten text (for marks detection)."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Red color ranges in HSV
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([160, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        red_mask = mask1 + mask2
        
        # Create white background with red text as black
        result = np.ones_like(frame) * 255
        result[red_mask > 0] = [0, 0, 0]
        
        return cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    
    def _ocr_loop(self) -> None:
        """Continuous OCR processing loop."""
        while not self._stop_event.is_set():
            if self.exchange is None or not self.exchange.is_active():
                time.sleep(0.1)
                continue
            
            try:
                grabbed, frame = self.exchange.read()
                
                if not grabbed or frame is None:
                    time.sleep(0.01)
                    continue
                
                # Crop to region of interest if specified
                if self.crop_region:
                    x, y, w, h = self.crop_region
                    frame = frame[y:y+h, x:x+w]
                
                # Preprocess
                processed = self._preprocess_frame(frame)
                
                # Perform OCR
                if self.use_easyocr and self._easyocr_reader:
                    results = self._easyocr_reader.readtext(processed)
                    # Convert EasyOCR format to similar format as Tesseract
                    boxes_data = self._convert_easyocr_results(results, frame.shape)
                    with self._lock:
                        self.boxes = boxes_data
                else:
                    # Tesseract OCR
                    boxes_data = pytesseract.image_to_data(
                        processed, 
                        lang=self.language,
                        output_type=pytesseract.Output.STRING
                    )
                    with self._lock:
                        self.boxes = boxes_data
                
            except Exception as e:
                print(f"OCR Error: {e}")
                time.sleep(0.1)
    
    def _convert_easyocr_results(self, results: List, frame_shape: Tuple) -> str:
        """Convert EasyOCR results to Tesseract-like format."""
        lines = ["level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext"]
        
        for i, (bbox, text, conf) in enumerate(results):
            # EasyOCR bbox: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
            x1, y1 = int(bbox[0][0]), int(bbox[0][1])
            x2, y2 = int(bbox[2][0]), int(bbox[2][1])
            w, h = x2 - x1, y2 - y1
            
            lines.append(f"5\t1\t1\t1\t{i+1}\t1\t{x1}\t{y1}\t{w}\t{h}\t{int(conf*100)}\t{text}")
        
        return "\n".join(lines)
    
    def get_results(self) -> Tuple[Optional[str], str, float]:
        """Get the latest OCR results."""
        with self._lock:
            return self.boxes, self.text, self.confidence
    
    def stop(self) -> None:
        """Stop the OCR processing thread."""
        self._stop_event.set()
        self.stopped = True
        
        if self._thread:
            self._thread.join(timeout=2.0)


class ViewMode:
    """View modes for OCR visualization."""
    
    CONFIDENCE_HIGH = 1      # Only high confidence (>75%)
    CONFIDENCE_COLOR = 2     # Green=high, Red=low confidence
    CONFIDENCE_GRADIENT = 3  # Color gradient based on confidence
    ALL_DETECTIONS = 4       # Show all detections
    
    @staticmethod
    def get_color_and_threshold(mode: int, confidence: float) -> Tuple[int, Tuple[int, int, int]]:
        """
        Get confidence threshold and color based on view mode.
        
        Args:
            mode: View mode (1-4)
            confidence: Detection confidence (0-100)
            
        Returns:
            Tuple of (threshold, BGR color)
        """
        if mode == 1:
            return 75, (0, 255, 0)  # Green, high confidence only
        
        elif mode == 2:
            if confidence >= 50:
                return 0, (0, 255, 0)  # Green for high
            else:
                return 0, (0, 0, 255)  # Red for low
        
        elif mode == 3:
            # Gradient: darker = lower confidence, brighter = higher
            intensity = int(confidence * 2.55)
            return 0, (intensity, intensity, 0)
        
        elif mode == 4:
            return 0, (0, 0, 255)  # Red, show all
        
        return 75, (0, 255, 0)  # Default


class RealTimeOCRStream:
    """
    Main class for real-time OCR streaming.
    
    Combines threaded video capture and OCR processing
    for seamless real-time text detection display.
    """
    
    def __init__(
        self,
        camera_src: int = 0,
        resolution: Tuple[int, int] = (1280, 720),
        crop: Optional[Tuple[int, int]] = None,
        view_mode: int = 1,
        language: str = 'eng',
        use_easyocr: bool = False
    ):
        """
        Initialize real-time OCR stream.
        
        Args:
            camera_src: Camera index
            resolution: Capture resolution (width, height)
            crop: Crop amount (width_crop, height_crop) - reduces OCR area
            view_mode: Display mode (1-4)
            language: OCR language code
            use_easyocr: Use EasyOCR instead of Tesseract
        """
        self.camera_src = camera_src
        self.resolution = resolution
        self.crop = crop or (0, 0)
        self.view_mode = view_mode
        self.language = language
        self.use_easyocr = use_easyocr
        
        # Components
        self.video_stream: Optional[ThreadedVideoStream] = None
        self.ocr_processor: Optional[ThreadedOCR] = None
        self.rate_counter: Optional[RateCounter] = None
        
        # State
        self.running = False
        self.captures = 0
        self.detected_text = ""
    
    def start(self) -> bool:
        """Start the real-time OCR stream."""
        try:
            # Start video stream
            self.video_stream = ThreadedVideoStream(
                src=self.camera_src,
                resolution=self.resolution
            ).start()
            
            width, height = self.video_stream.get_dimensions()
            print(f"Video stream started: {width}x{height}")
            
            # Start OCR processor
            self.ocr_processor = ThreadedOCR(
                use_easyocr=self.use_easyocr,
                language=self.language
            )
            self.ocr_processor.set_video_stream(self.video_stream)
            
            # Set crop region if specified
            if self.crop[0] > 0 or self.crop[1] > 0:
                crop_x, crop_y = self.crop
                self.ocr_processor.set_crop_region(
                    crop_x, crop_y,
                    width - 2*crop_x, height - 2*crop_y
                )
            
            self.ocr_processor.start()
            print("OCR processor started")
            
            # Start rate counter
            self.rate_counter = RateCounter().start()
            
            self.running = True
            return True
            
        except Exception as e:
            print(f"Error starting OCR stream: {e}")
            self.stop()
            return False
    
    def stop(self) -> None:
        """Stop the real-time OCR stream."""
        self.running = False
        
        if self.ocr_processor:
            self.ocr_processor.stop()
            self.ocr_processor = None
        
        if self.video_stream:
            self.video_stream.stop()
            self.video_stream = None
        
        print(f"OCR stream stopped. {self.captures} image(s) captured.")
    
    def _draw_ocr_boxes(
        self,
        frame: np.ndarray,
        boxes_data: Optional[str],
        height: int,
        crop_offset: Tuple[int, int] = (0, 0)
    ) -> Tuple[np.ndarray, str]:
        """
        Draw OCR bounding boxes on frame.
        
        Args:
            frame: Display frame
            boxes_data: Tesseract image_to_data output
            height: Frame height
            crop_offset: (x, y) offset for cropped OCR
            
        Returns:
            Frame with boxes drawn and detected text
        """
        detected_text = ""
        
        if boxes_data is None:
            return frame, detected_text
        
        for i, line in enumerate(boxes_data.splitlines()):
            if i == 0:  # Skip header
                continue
            
            parts = line.split('\t')
            if len(parts) >= 12:
                try:
                    x = int(parts[6]) + crop_offset[0]
                    y = int(parts[7]) + crop_offset[1]
                    w = int(parts[8])
                    h = int(parts[9])
                    conf = float(parts[10])
                    text = parts[11]
                    
                    # Get color based on view mode
                    threshold, color = ViewMode.get_color_and_threshold(self.view_mode, conf)
                    
                    if conf > threshold:
                        # Draw bounding box
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 1)
                        detected_text += " " + text
                        
                except (ValueError, IndexError):
                    continue
        
        # Display detected text at bottom
        if detected_text.strip() and detected_text.isascii():
            cv2.putText(
                frame, detected_text.strip()[:100],  # Limit length
                (5, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1
            )
        
        self.detected_text = detected_text.strip()
        return frame, detected_text
    
    def _draw_overlay(self, frame: np.ndarray, width: int, height: int) -> np.ndarray:
        """Draw overlay information on frame."""
        # Frame rate
        if self.rate_counter:
            fps = int(self.rate_counter.rate())
            cv2.putText(
                frame, f"FPS: {fps}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )
        
        # View mode
        mode_names = {
            1: "High Conf",
            2: "Color Code",
            3: "Gradient",
            4: "All Text"
        }
        cv2.putText(
            frame, f"Mode: {mode_names.get(self.view_mode, '?')}",
            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )
        
        # Language
        cv2.putText(
            frame, f"Lang: {self.language}",
            (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )
        
        # Crop box
        if self.crop[0] > 0 or self.crop[1] > 0:
            cx, cy = self.crop
            cv2.rectangle(
                frame, (cx, cy), (width - cx, height - cy),
                (255, 0, 0), 2
            )
        
        # Instructions
        cv2.putText(
            frame, "Press 'c' to capture | 'q' to quit | 1-4 change view",
            (10, height - 35),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
        )
        
        return frame
    
    def run_display(self, window_name: str = "Real-Time OCR") -> None:
        """
        Run the real-time OCR display loop.
        
        This is for standalone desktop usage with OpenCV window.
        """
        if not self.start():
            return
        
        width, height = self.video_stream.get_dimensions()
        
        print(f"\n{'='*50}")
        print("Real-Time OCR Started")
        print(f"Resolution: {width}x{height}")
        print(f"View Mode: {self.view_mode}")
        print(f"Language: {self.language}")
        print("Press 'c' to capture image, 'q' to quit")
        print(f"{'='*50}\n")
        
        try:
            while self.running:
                # Get frame
                grabbed, frame = self.video_stream.read()
                
                if not grabbed or frame is None:
                    time.sleep(0.01)
                    continue
                
                # Draw OCR boxes
                crop_offset = (self.crop[0], self.crop[1])
                frame, _ = self._draw_ocr_boxes(
                    frame,
                    self.ocr_processor.boxes,
                    height,
                    crop_offset
                )
                
                # Draw overlay
                frame = self._draw_overlay(frame, width, height)
                
                # Display
                cv2.imshow(window_name, frame)
                
                if self.rate_counter:
                    self.rate_counter.increment()
                
                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self._capture_image(frame)
                elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:
                    self.view_mode = key - ord('0')
                    print(f"View mode changed to {self.view_mode}")
        
        finally:
            cv2.destroyAllWindows()
            self.stop()
    
    def _capture_image(self, frame: np.ndarray) -> str:
        """Capture and save current frame."""
        # Create images folder
        images_dir = Path("./images")
        images_dir.mkdir(exist_ok=True)
        
        # Generate filename
        now = datetime.now()
        filename = f"OCR_{now.strftime('%Y-%m-%d_at_%H-%M-%S')}_{self.captures + 1}.jpg"
        filepath = images_dir / filename
        
        # Save image
        cv2.imwrite(str(filepath), frame)
        self.captures += 1
        
        print(f"\nCaptured: {filename}")
        print(f"Text: {self.detected_text[:200]}..." if len(self.detected_text) > 200 else f"Text: {self.detected_text}")
        
        return str(filepath)
    
    def get_frame_with_ocr(self) -> Tuple[bool, Optional[np.ndarray], str]:
        """
        Get current frame with OCR overlay.
        
        For web streaming integration.
        
        Returns:
            Tuple of (success, frame, detected_text)
        """
        if not self.running or not self.video_stream:
            return False, None, ""
        
        grabbed, frame = self.video_stream.read()
        if not grabbed or frame is None:
            return False, None, ""
        
        width, height = self.video_stream.get_dimensions()
        
        # Draw OCR boxes
        crop_offset = (self.crop[0], self.crop[1])
        frame, text = self._draw_ocr_boxes(
            frame,
            self.ocr_processor.boxes if self.ocr_processor else None,
            height,
            crop_offset
        )
        
        # Draw overlay
        frame = self._draw_overlay(frame, width, height)
        
        if self.rate_counter:
            self.rate_counter.increment()
        
        return True, frame, text
    
    def get_jpeg_frame_with_ocr(self, quality: int = 85) -> Optional[bytes]:
        """
        Get JPEG-encoded frame with OCR overlay.
        
        For web streaming.
        
        Returns:
            JPEG bytes or None
        """
        success, frame, _ = self.get_frame_with_ocr()
        
        if not success or frame is None:
            return None
        
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        
        if ret:
            return buffer.tobytes()
        return None
    
    def generate_frames_with_ocr(self):
        """
        Generator for streaming frames with OCR overlay.
        
        Yields MJPEG frames for web streaming.
        """
        while self.running:
            jpeg_bytes = self.get_jpeg_frame_with_ocr()
            
            if jpeg_bytes:
                yield (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + jpeg_bytes + b'\r\n'
                )
            
            time.sleep(0.033)  # ~30 FPS


# Singleton instance for web app integration
_realtime_ocr_instance: Optional[RealTimeOCRStream] = None


def get_realtime_ocr(
    camera_src: int = 0,
    resolution: Tuple[int, int] = (1280, 720),
    crop: Optional[Tuple[int, int]] = None,
    view_mode: int = 1,
    language: str = 'eng',
    use_easyocr: bool = False,
    force_new: bool = False
) -> RealTimeOCRStream:
    """
    Get or create singleton RealTimeOCRStream instance.
    
    Args:
        camera_src: Camera index
        resolution: Capture resolution
        crop: Crop amount for OCR region
        view_mode: Display mode (1-4)
        language: OCR language
        use_easyocr: Use EasyOCR
        force_new: Force create new instance
        
    Returns:
        RealTimeOCRStream instance
    """
    global _realtime_ocr_instance
    
    if force_new and _realtime_ocr_instance:
        _realtime_ocr_instance.stop()
        _realtime_ocr_instance = None
    
    if _realtime_ocr_instance is None:
        _realtime_ocr_instance = RealTimeOCRStream(
            camera_src=camera_src,
            resolution=resolution,
            crop=crop,
            view_mode=view_mode,
            language=language,
            use_easyocr=use_easyocr
        )
    
    return _realtime_ocr_instance


def stop_realtime_ocr():
    """Stop and cleanup singleton instance."""
    global _realtime_ocr_instance
    
    if _realtime_ocr_instance:
        _realtime_ocr_instance.stop()
        _realtime_ocr_instance = None


# ==================== Main Entry Point ====================

def main():
    """Run standalone real-time OCR."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-Time OCR with Webcam")
    parser.add_argument('-s', '--src', type=int, default=0,
                        help="Camera source index (default: 0)")
    parser.add_argument('-c', '--crop', type=int, nargs=2, default=[100, 100],
                        metavar=('W', 'H'),
                        help="Crop amount in pixels: width height")
    parser.add_argument('-v', '--view', type=int, default=1, choices=[1, 2, 3, 4],
                        help="View mode (1-4)")
    parser.add_argument('-l', '--lang', type=str, default='eng',
                        help="OCR language code (e.g., 'eng', 'chi_sim+eng')")
    parser.add_argument('--easyocr', action='store_true',
                        help="Use EasyOCR instead of Tesseract")
    parser.add_argument('-r', '--resolution', type=int, nargs=2, default=[1280, 720],
                        metavar=('W', 'H'),
                        help="Camera resolution: width height")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("PRODUCTION REAL-TIME OCR")
    print("="*60)
    print(f"Camera: {args.src}")
    print(f"Resolution: {args.resolution[0]}x{args.resolution[1]}")
    print(f"Crop: {args.crop}")
    print(f"View Mode: {args.view}")
    print(f"Language: {args.lang}")
    print(f"OCR Engine: {'EasyOCR' if args.easyocr else 'Tesseract'}")
    print("="*60 + "\n")
    
    # Check OCR availability
    if not PYTESSERACT_AVAILABLE and not args.easyocr:
        print("ERROR: Tesseract not available. Install pytesseract or use --easyocr flag.")
        sys.exit(1)
    
    if args.easyocr and not EASYOCR_AVAILABLE:
        print("WARNING: EasyOCR not available. Falling back to Tesseract.")
        args.easyocr = False
    
    # Run OCR stream
    ocr_stream = RealTimeOCRStream(
        camera_src=args.src,
        resolution=tuple(args.resolution),
        crop=tuple(args.crop),
        view_mode=args.view,
        language=args.lang,
        use_easyocr=args.easyocr
    )
    
    ocr_stream.run_display()


if __name__ == "__main__":
    main()
