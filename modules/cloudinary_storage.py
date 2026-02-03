"""
Cloudinary Storage Module
Handles image upload and management using Cloudinary cloud storage.
"""
import os
import io
import cv2
import numpy as np
from datetime import datetime
from typing import Optional, Tuple, Dict, Any

# Cloudinary imports
CLOUDINARY_AVAILABLE = False
try:
    import cloudinary
    import cloudinary.uploader
    import cloudinary.api
    from cloudinary.utils import cloudinary_url
    CLOUDINARY_AVAILABLE = True
except ImportError:
    print("Cloudinary not installed. Install with: pip install cloudinary")


class CloudinaryStorage:
    """
    Cloudinary cloud storage manager for OCR images.
    
    Features:
    - Upload images directly from OpenCV frames
    - Upload from file paths
    - Upload from bytes
    - Automatic folder organization
    - Image transformation options
    """
    
    def __init__(
        self,
        cloud_name: str = None,
        api_key: str = None,
        api_secret: str = None
    ):
        """
        Initialize Cloudinary storage.
        
        Args:
            cloud_name: Cloudinary cloud name
            api_key: Cloudinary API key
            api_secret: Cloudinary API secret
        """
        self.configured = False
        
        if not CLOUDINARY_AVAILABLE:
            print("Warning: Cloudinary package not available")
            return
        
        # Get credentials from parameters or environment
        self.cloud_name = cloud_name or os.getenv('CLOUDINARY_CLOUD_NAME', '')
        self.api_key = api_key or os.getenv('CLOUDINARY_API_KEY', '')
        self.api_secret = api_secret or os.getenv('CLOUDINARY_API_SECRET', '')
        
        if self.cloud_name and self.api_key and self.api_secret:
            self._configure()
        else:
            print("Warning: Cloudinary credentials not provided")
    
    def _configure(self) -> None:
        """Configure Cloudinary with credentials."""
        try:
            cloudinary.config(
                cloud_name=self.cloud_name,
                api_key=self.api_key,
                api_secret=self.api_secret,
                secure=True
            )
            self.configured = True
            print("✓ Cloudinary configured successfully")
        except Exception as e:
            print(f"✗ Cloudinary configuration failed: {e}")
            self.configured = False
    
    def is_available(self) -> bool:
        """Check if Cloudinary is available and configured."""
        return CLOUDINARY_AVAILABLE and self.configured
    
    def upload_frame(
        self,
        frame: np.ndarray,
        folder: str = "ocr_captures",
        prefix: str = "capture",
        tags: list = None,
        **options
    ) -> Tuple[bool, Optional[str], str]:
        """
        Upload an OpenCV frame to Cloudinary.
        
        Args:
            frame: OpenCV BGR image array
            folder: Cloudinary folder name
            prefix: Filename prefix
            tags: List of tags for the image
            **options: Additional Cloudinary upload options
            
        Returns:
            Tuple of (success, url, message)
        """
        if not self.is_available():
            return False, None, "Cloudinary not available"
        
        try:
            # Encode frame to JPEG bytes
            success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not success:
                return False, None, "Failed to encode image"
            
            # Create bytes IO
            image_bytes = io.BytesIO(buffer.tobytes())
            
            # Generate public_id with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            public_id = f"{folder}/{prefix}_{timestamp}"
            
            # Upload options
            upload_options = {
                'public_id': public_id,
                'folder': None,  # Already included in public_id
                'resource_type': 'image',
                'tags': tags or ['ocr', 'capture'],
                'overwrite': True,
                **options
            }
            
            # Upload to Cloudinary
            result = cloudinary.uploader.upload(image_bytes, **upload_options)
            
            url = result.get('secure_url') or result.get('url')
            return True, url, "Image uploaded successfully"
            
        except Exception as e:
            return False, None, f"Upload failed: {str(e)}"
    
    def upload_file(
        self,
        file_path: str,
        folder: str = "ocr_uploads",
        tags: list = None,
        **options
    ) -> Tuple[bool, Optional[str], str]:
        """
        Upload an image file to Cloudinary.
        
        Args:
            file_path: Path to the image file
            folder: Cloudinary folder name
            tags: List of tags for the image
            **options: Additional Cloudinary upload options
            
        Returns:
            Tuple of (success, url, message)
        """
        if not self.is_available():
            return False, None, "Cloudinary not available"
        
        if not os.path.exists(file_path):
            return False, None, f"File not found: {file_path}"
        
        try:
            # Generate public_id from filename
            filename = os.path.basename(file_path)
            name, _ = os.path.splitext(filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            public_id = f"{folder}/{name}_{timestamp}"
            
            # Upload options
            upload_options = {
                'public_id': public_id,
                'resource_type': 'image',
                'tags': tags or ['ocr', 'upload'],
                'overwrite': True,
                **options
            }
            
            # Upload to Cloudinary
            result = cloudinary.uploader.upload(file_path, **upload_options)
            
            url = result.get('secure_url') or result.get('url')
            return True, url, "Image uploaded successfully"
            
        except Exception as e:
            return False, None, f"Upload failed: {str(e)}"
    
    def upload_bytes(
        self,
        image_bytes: bytes,
        folder: str = "ocr_uploads",
        prefix: str = "upload",
        tags: list = None,
        **options
    ) -> Tuple[bool, Optional[str], str]:
        """
        Upload image bytes to Cloudinary.
        
        Args:
            image_bytes: Raw image bytes
            folder: Cloudinary folder name
            prefix: Filename prefix
            tags: List of tags for the image
            **options: Additional Cloudinary upload options
            
        Returns:
            Tuple of (success, url, message)
        """
        if not self.is_available():
            return False, None, "Cloudinary not available"
        
        try:
            # Generate public_id with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            public_id = f"{folder}/{prefix}_{timestamp}"
            
            # Upload options
            upload_options = {
                'public_id': public_id,
                'resource_type': 'image',
                'tags': tags or ['ocr', 'upload'],
                'overwrite': True,
                **options
            }
            
            # Upload to Cloudinary
            result = cloudinary.uploader.upload(
                io.BytesIO(image_bytes),
                **upload_options
            )
            
            url = result.get('secure_url') or result.get('url')
            return True, url, "Image uploaded successfully"
            
        except Exception as e:
            return False, None, f"Upload failed: {str(e)}"
    
    def get_optimized_url(
        self,
        public_id: str,
        width: int = None,
        height: int = None,
        crop: str = "fill",
        quality: str = "auto",
        format: str = "auto"
    ) -> str:
        """
        Get optimized URL for an image with transformations.
        
        Args:
            public_id: Cloudinary public ID
            width: Desired width
            height: Desired height
            crop: Crop mode
            quality: Quality setting
            format: Output format
            
        Returns:
            Optimized image URL
        """
        if not self.is_available():
            return ""
        
        transformations = {
            'quality': quality,
            'fetch_format': format
        }
        
        if width:
            transformations['width'] = width
        if height:
            transformations['height'] = height
        if width or height:
            transformations['crop'] = crop
        
        url, _ = cloudinary_url(public_id, **transformations)
        return url
    
    def delete_image(self, public_id: str) -> Tuple[bool, str]:
        """
        Delete an image from Cloudinary.
        
        Args:
            public_id: Cloudinary public ID
            
        Returns:
            Tuple of (success, message)
        """
        if not self.is_available():
            return False, "Cloudinary not available"
        
        try:
            result = cloudinary.uploader.destroy(public_id)
            if result.get('result') == 'ok':
                return True, "Image deleted successfully"
            else:
                return False, f"Delete failed: {result}"
        except Exception as e:
            return False, f"Delete failed: {str(e)}"
    
    def list_images(
        self,
        folder: str = "ocr_captures",
        max_results: int = 50
    ) -> Tuple[bool, list, str]:
        """
        List images in a Cloudinary folder.
        
        Args:
            folder: Folder to list
            max_results: Maximum number of results
            
        Returns:
            Tuple of (success, images_list, message)
        """
        if not self.is_available():
            return False, [], "Cloudinary not available"
        
        try:
            result = cloudinary.api.resources(
                type='upload',
                prefix=folder,
                max_results=max_results
            )
            
            images = []
            for resource in result.get('resources', []):
                images.append({
                    'public_id': resource.get('public_id'),
                    'url': resource.get('secure_url'),
                    'format': resource.get('format'),
                    'width': resource.get('width'),
                    'height': resource.get('height'),
                    'created_at': resource.get('created_at')
                })
            
            return True, images, f"Found {len(images)} images"
            
        except Exception as e:
            return False, [], f"List failed: {str(e)}"


# Singleton instance
_cloudinary_storage = None


def get_cloudinary_storage() -> CloudinaryStorage:
    """Get or create singleton CloudinaryStorage instance."""
    global _cloudinary_storage
    if _cloudinary_storage is None:
        _cloudinary_storage = CloudinaryStorage()
    return _cloudinary_storage


# Helper function for quick uploads
def upload_to_cloudinary(
    image,  # Can be np.ndarray, str (file path), or bytes
    folder: str = "ocr_captures",
    tags: list = None
) -> Tuple[bool, Optional[str], str]:
    """
    Quick helper to upload any image type to Cloudinary.
    
    Args:
        image: OpenCV frame, file path, or bytes
        folder: Cloudinary folder
        tags: Image tags
        
    Returns:
        Tuple of (success, url, message)
    """
    storage = get_cloudinary_storage()
    
    if isinstance(image, np.ndarray):
        return storage.upload_frame(image, folder=folder, tags=tags)
    elif isinstance(image, str):
        return storage.upload_file(image, folder=folder, tags=tags)
    elif isinstance(image, bytes):
        return storage.upload_bytes(image, folder=folder, tags=tags)
    else:
        return False, None, f"Unsupported image type: {type(image)}"


# Test the module
if __name__ == "__main__":
    print("\nTesting Cloudinary Storage...")
    
    storage = get_cloudinary_storage()
    
    if storage.is_available():
        print("✓ Cloudinary is available")
        
        # Test with a simple image
        test_image = np.zeros((100, 200, 3), dtype=np.uint8)
        cv2.putText(test_image, "TEST", (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        success, url, message = storage.upload_frame(test_image, folder="ocr_test", prefix="test")
        print(f"Upload test: {message}")
        if success:
            print(f"URL: {url}")
    else:
        print("✗ Cloudinary is not available")
