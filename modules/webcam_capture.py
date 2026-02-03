"""
Webcam Capture Module - Production Level
Handles webcam operations and frame capture with optimized threading.

Features:
- Threaded frame capture for non-blocking operation
- Real-time OCR overlay support
- Multiple preprocessing modes
- Frame rate monitoring
- Graceful error handling
"""
import cv2
import numpy as np
import time
import threading
from threading import Thread, Event
from datetime import datetime
import os
from typing import Optional, Tuple, Generator

from config import Config


class FrameRateCounter:
    """Thread-safe frame rate counter."""
    
    def __init__(self, sample_size: int = 30):
        self.sample_size = sample_size
        self.timestamps = []
        self._lock = threading.Lock()
    
    def tick(self) -> None:
        """Record a frame timestamp."""
        with self._lock:
            now = time.perf_counter()
            self.timestamps.append(now)
            # Keep only recent samples
            if len(self.timestamps) > self.sample_size:
                self.timestamps.pop(0)
    
    def get_fps(self) -> float:
        """Calculate current FPS."""
        with self._lock:
            if len(self.timestamps) < 2:
                return 0.0
            elapsed = self.timestamps[-1] - self.timestamps[0]
            if elapsed <= 0:
                return 0.0
            return (len(self.timestamps) - 1) / elapsed


class WebcamManager:
    """
    Production-level webcam manager with optimized threading.
    
    Features:
    - Non-blocking frame capture in dedicated thread
    - Thread-safe frame access
    - Configurable resolution and FPS
    - Real-time FPS monitoring
    - Graceful shutdown handling
    """
    
    def __init__(self):
        self.camera: Optional[cv2.VideoCapture] = None
        self.is_running: bool = False
        self.current_frame: Optional[np.ndarray] = None
        self.frame_count: int = 0
        
        # Thread synchronization
        self.lock = threading.Lock()
        self.capture_thread: Optional[Thread] = None
        self._stop_event = Event()
        
        # Performance monitoring
        self.fps_counter = FrameRateCounter()
        self.last_frame_time: float = 0.0
        
        # Frame dimensions
        self.frame_width: int = 0
        self.frame_height: int = 0
        
    def start(self, camera_index: Optional[int] = None) -> Tuple[bool, str]:
        """
        Start the webcam capture.
        
        Args:
            camera_index: Camera device index (default from config)
            
        Returns:
            Tuple of (success, message)
        """
        if self.is_running:
            return True, "Camera already running"
        
        try:
            index = camera_index if camera_index is not None else Config.CAMERA_INDEX
            
            # Try DirectShow first on Windows for better performance
            self.camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            
            if not self.camera.isOpened():
                # Fallback without DirectShow
                self.camera = cv2.VideoCapture(index)
            
            if not self.camera.isOpened():
                return False, "Failed to open camera"
            
            # Configure camera settings
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAPTURE_WIDTH)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAPTURE_HEIGHT)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
            self.camera.set(cv2.CAP_PROP_FPS, 30)  # Request 30 FPS
            
            # Get actual dimensions
            self.frame_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Read first frame to verify
            ret, frame = self.camera.read()
            if not ret:
                self.camera.release()
                return False, "Failed to read from camera"
            
            with self.lock:
                self.current_frame = frame
            
            self.is_running = True
            self._stop_event.clear()
            self.frame_count = 0
            
            # Start capture thread
            self.capture_thread = Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            return True, f"Camera started: {self.frame_width}x{self.frame_height}"
            
        except Exception as e:
            if self.camera:
                self.camera.release()
                self.camera = None
            return False, f"Error starting camera: {str(e)}"
    
    def stop(self) -> Tuple[bool, str]:
        """Stop the webcam capture gracefully."""
        if not self.is_running:
            return True, "Camera not running"
        
        self._stop_event.set()
        self.is_running = False
        
        # Wait for capture thread to finish
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        
        # Release camera
        if self.camera:
            self.camera.release()
            self.camera = None
        
        return True, f"Camera stopped. {self.frame_count} frames captured."
    
    def _capture_loop(self) -> None:
        """
        Continuous frame capture loop running in dedicated thread.
        
        Optimized for minimal latency and consistent frame timing.
        """
        target_interval = 1.0 / 30.0  # Target 30 FPS
        
        while not self._stop_event.is_set():
            if self.camera is None or not self.camera.isOpened():
                break
            
            loop_start = time.perf_counter()
            
            ret, frame = self.camera.read()
            
            if ret and frame is not None:
                with self.lock:
                    self.current_frame = frame
                    self.frame_count += 1
                    self.last_frame_time = time.perf_counter()
                
                self.fps_counter.tick()
            
            # Maintain consistent frame rate
            elapsed = time.perf_counter() - loop_start
            sleep_time = max(0.001, target_interval - elapsed)
            time.sleep(sleep_time)
    
    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Get the current frame (thread-safe).
        
        Returns:
            Tuple of (success, frame_copy)
        """
        with self.lock:
            if self.current_frame is not None:
                return True, self.current_frame.copy()
            return False, None
    
    def get_dimensions(self) -> Tuple[int, int]:
        """Get frame dimensions (width, height)."""
        return self.frame_width, self.frame_height
    
    def get_fps(self) -> float:
        """Get current frames per second."""
        return self.fps_counter.get_fps()
    
    def capture_image(self, save_path: Optional[str] = None) -> Tuple[bool, Optional[np.ndarray], str]:
        """
        Capture and optionally save the current frame.
        
        Args:
            save_path: Optional path to save the image
            
        Returns:
            Tuple of (success, frame, message)
        """
        success, frame = self.get_frame()
        
        if not success or frame is None:
            return False, None, "No frame available"
        
        if save_path:
            cv2.imwrite(save_path, frame)
            return True, frame, f"Image saved to {save_path}"
        
        return True, frame, "Image captured"
    
    def generate_frames(self, with_overlay: bool = True, jpeg_quality: int = 85) -> Generator:
        """
        Generator for streaming frames (for web interface).
        
        Args:
            with_overlay: Add timestamp and instruction overlay
            jpeg_quality: JPEG compression quality (0-100)
            
        Yields:
            MJPEG frame bytes
        """
        while self.is_running:
            success, frame = self.get_frame()
            
            if success and frame is not None:
                if with_overlay:
                    frame = self._add_overlay(frame)
                
                # Encode frame as JPEG
                ret, buffer = cv2.imencode(
                    '.jpg', frame,
                    [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
                )
                
                if ret:
                    yield (
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n'
                    )
            
            time.sleep(0.033)  # ~30 FPS
    
    def _add_overlay(self, frame: np.ndarray) -> np.ndarray:
        """Add informational overlay to frame."""
        # Timestamp
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cv2.putText(
            frame, timestamp, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
        )
        
        # FPS
        fps = int(self.get_fps())
        cv2.putText(
            frame, f"FPS: {fps}", (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
        )
        
        # Frame count
        cv2.putText(
            frame, f"Frame: {self.frame_count}", (10, 85),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
        )
        
        # Instruction text
        cv2.putText(
            frame, "Press SPACE or Click 'Capture' to scan document",
            (10, frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
        )
        
        return frame
    
    def get_jpeg_frame(self, with_overlay: bool = True, quality: int = 90) -> Optional[bytes]:
        """
        Get current frame as JPEG bytes.
        
        Args:
            with_overlay: Add timestamp overlay
            quality: JPEG quality (0-100)
            
        Returns:
            JPEG bytes or None
        """
        success, frame = self.get_frame()
        
        if success and frame is not None:
            if with_overlay:
                frame = self._add_overlay(frame)
            
            # Encode as JPEG
            ret, buffer = cv2.imencode(
                '.jpg', frame,
                [cv2.IMWRITE_JPEG_QUALITY, quality]
            )
            
            if ret:
                return buffer.tobytes()
        
        return None
    
    def is_active(self) -> bool:
        """Check if camera is active and capturing."""
        return (
            self.is_running and 
            self.camera is not None and 
            self.camera.isOpened()
        )


class ImageCapture:
    """Static methods for one-time image capture and processing."""
    
    @staticmethod
    def capture_from_camera(
        camera_index: int = 0,
        save_path: Optional[str] = None,
        warmup_frames: int = 5
    ) -> Tuple[bool, Optional[np.ndarray], str]:
        """
        Capture a single image from camera.
        
        Args:
            camera_index: Camera device index
            save_path: Optional path to save image
            warmup_frames: Number of frames to skip for camera warmup
            
        Returns:
            Tuple of (success, image, message)
        """
        cap = None
        try:
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
            
            if not cap.isOpened():
                cap = cv2.VideoCapture(camera_index)
            
            if not cap.isOpened():
                return False, None, "Could not open camera"
            
            # Set resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.CAPTURE_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.CAPTURE_HEIGHT)
            
            # Warm up camera (discard initial frames)
            for _ in range(warmup_frames):
                cap.read()
            
            # Capture frame
            ret, frame = cap.read()
            
            if not ret or frame is None:
                return False, None, "Failed to capture image"
            
            if save_path:
                cv2.imwrite(save_path, frame)
            
            return True, frame, "Image captured successfully"
            
        except Exception as e:
            return False, None, f"Error capturing image: {str(e)}"
        
        finally:
            if cap:
                cap.release()
    
    @staticmethod
    def load_image(file_path: str) -> Tuple[bool, Optional[np.ndarray], str]:
        """
        Load an image from file.
        
        Args:
            file_path: Path to image file
            
        Returns:
            Tuple of (success, image, message)
        """
        if not os.path.exists(file_path):
            return False, None, "File not found"
        
        try:
            image = cv2.imread(file_path)
            if image is None:
                return False, None, "Failed to load image"
            
            return True, image, "Image loaded successfully"
            
        except Exception as e:
            return False, None, f"Error loading image: {str(e)}"
    
    @staticmethod
    def image_from_bytes(image_bytes: bytes) -> Tuple[bool, Optional[np.ndarray], str]:
        """
        Convert image bytes to OpenCV image.
        
        Args:
            image_bytes: Raw image bytes
            
        Returns:
            Tuple of (success, image, message)
        """
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return False, None, "Failed to decode image"
            
            return True, image, "Image decoded successfully"
            
        except Exception as e:
            return False, None, f"Error decoding image: {str(e)}"
    
    @staticmethod
    def preprocess_for_ocr(
        image: np.ndarray,
        enhance_contrast: bool = True,
        denoise: bool = True
    ) -> np.ndarray:
        """
        Preprocess image for better OCR accuracy.
        
        Args:
            image: Input BGR image
            enhance_contrast: Apply CLAHE contrast enhancement
            denoise: Apply noise reduction
            
        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Denoise
        if denoise:
            gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # Enhance contrast
        if enhance_contrast:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        
        return gray


# Singleton webcam manager
_webcam_manager = None

def get_webcam_manager():
    """Get or create the webcam manager instance"""
    global _webcam_manager
    if _webcam_manager is None:
        _webcam_manager = WebcamManager()
    return _webcam_manager


# Test the module
if __name__ == "__main__":
    print("Testing webcam capture...")
    
    manager = get_webcam_manager()
    success, message = manager.start()
    print(f"Start camera: {message}")
    
    if success:
        print("Camera started. Press 'q' to quit, 's' to save image...")
        
        while True:
            success, frame = manager.get_frame()
            
            if success and frame is not None:
                cv2.imshow('Webcam Test', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                Config.ensure_upload_folder()
                save_path = os.path.join(Config.UPLOAD_FOLDER, f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg")
                success, _, msg = manager.capture_image(save_path)
                print(msg)
        
        cv2.destroyAllWindows()
        manager.stop()
        print("Camera stopped.")
    else:
        print("Failed to start camera")
