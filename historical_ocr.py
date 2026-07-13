"""Specialized OCR for historical manuscripts and artifacts."""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from .config import Config


class HistoricalOCR:
    """
    OCR optimized for historical documents, manuscripts, and inscriptions.
    Includes preprocessing for degraded/ancient text.
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.reader = None
        self.backend = "easyocr"  # Better for historical text than Tesseract
        self.load_model()
    
    def load_model(self):
        """Load OCR model optimized for historical text."""
        try:
            import easyocr
            # Support English and Latin
            languages = ['en']  # Start with English only
            self.reader = easyocr.Reader(
                languages,
                gpu=self.config.DEVICE == "cuda"
            )
            print(f"✓ Historical OCR loaded (languages: {', '.join(languages)})")
        except ImportError:
            print("⚠️  EasyOCR not installed. Install with: pip install easyocr")
            self.reader = None
    
    def preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Advanced preprocessing for historical documents.
        Handles degraded text, stains, uneven lighting.
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # 1. Denoise
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # 2. Enhance contrast using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # 3. Adaptive thresholding for uneven lighting
        binary = cv2.adaptiveThreshold(
            enhanced, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11, 2
        )
        
        # 4. Morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        
        return cleaned
    
    def preprocess_inscription(self, image: np.ndarray) -> np.ndarray:
        """
        Specialized preprocessing for stone inscriptions.
        Enhances carved/engraved text.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Edge enhancement for carved text
        edges = cv2.Canny(gray, 50, 150)
        
        # Dilate edges to make text more visible
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Combine with original
        enhanced = cv2.addWeighted(gray, 0.7, dilated, 0.3, 0)
        
        # Apply CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        result = clahe.apply(enhanced.astype(np.uint8))
        
        return result
    
    def extract_text_with_boxes(self, image: np.ndarray, 
                                document_type: str = 'manuscript',
                                min_confidence: Optional[float] = None) -> List[Dict]:
        """
        Extract text with bounding boxes from historical documents.
        
        Args:
            image: Input image
            document_type: 'manuscript', 'inscription', or 'papyrus'
            min_confidence: Minimum confidence threshold
        
        Returns:
            List of: [{'text': str, 'bbox': [x1, y1, x2, y2], 'confidence': float}]
        """
        if self.reader is None:
            return []
        
        if min_confidence is None:
            min_confidence = 0.3  # Lower threshold for historical text
        
        # Preprocess based on document type
        if document_type == 'inscription':
            preprocessed = self.preprocess_inscription(image)
        else:
            preprocessed = self.preprocess_for_ocr(image)
        
        # Run OCR
        try:
            results = self.reader.readtext(preprocessed)
            
            extracted = []
            for detection in results:
                bbox, text, confidence = detection
                
                if confidence >= min_confidence:
                    # Convert bbox to [x1, y1, x2, y2]
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    x1, y1 = min(x_coords), min(y_coords)
                    x2, y2 = max(x_coords), max(y_coords)
                    
                    extracted.append({
                        'text': text,
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(confidence),
                        'type': document_type
                    })
            
            return extracted
            
        except Exception as e:
            print(f"OCR error: {e}")
            return []
    
    def recognize_text(self, image: np.ndarray, document_type: str = 'manuscript') -> str:
        """
        Extract all text from image as a single string.
        
        Args:
            image: Input image
            document_type: Type of document for preprocessing
        
        Returns:
            Extracted text string
        """
        results = self.extract_text_with_boxes(image, document_type)
        
        if not results:
            return ""
        
        # Sort by vertical position (top to bottom)
        results.sort(key=lambda x: x['bbox'][1])
        
        # Combine text
        text_lines = [r['text'] for r in results]
        return " ".join(text_lines)
    
    def analyze_text_quality(self, image: np.ndarray) -> Dict:
        """
        Analyze text quality and readability.
        Helps determine if preprocessing is needed.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        
        # Calculate metrics
        contrast = gray.std()
        brightness = gray.mean()
        
        # Detect blur using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Estimate noise
        noise = self._estimate_noise(gray)
        
        quality = {
            'contrast': float(contrast),
            'brightness': float(brightness),
            'sharpness': float(laplacian_var),
            'noise_level': float(noise),
            'needs_preprocessing': contrast < 50 or laplacian_var < 100 or noise > 10
        }
        
        return quality
    
    def _estimate_noise(self, image: np.ndarray) -> float:
        """Estimate noise level in image."""
        h, w = image.shape
        
        # Use median absolute deviation
        median = np.median(image)
        mad = np.median(np.abs(image - median))
        
        return mad * 1.4826  # Scale factor for normal distribution
    
    def extract_with_confidence_analysis(self, image: np.ndarray, 
                                        document_type: str = 'manuscript') -> Dict:
        """
        Extract text with detailed confidence analysis.
        
        Returns:
            Dict with text, confidence stats, and quality metrics
        """
        # Analyze image quality
        quality = self.analyze_text_quality(image)
        
        # Extract text
        results = self.extract_text_with_boxes(image, document_type)
        
        if not results:
            return {
                'text': '',
                'word_count': 0,
                'avg_confidence': 0.0,
                'quality': quality,
                'readable': False
            }
        
        # Calculate statistics
        confidences = [r['confidence'] for r in results]
        text = " ".join([r['text'] for r in results])
        
        return {
            'text': text,
            'word_count': len(results),
            'avg_confidence': float(np.mean(confidences)),
            'min_confidence': float(np.min(confidences)),
            'max_confidence': float(np.max(confidences)),
            'quality': quality,
            'readable': np.mean(confidences) > 0.5,
            'details': results
        }
    
    def draw_text_boxes(self, image: np.ndarray, text_results: List[Dict]) -> np.ndarray:
        """Draw text bounding boxes with confidence colors."""
        img_draw = image.copy()
        
        for result in text_results:
            x1, y1, x2, y2 = map(int, result['bbox'])
            text = result['text']
            confidence = result['confidence']
            
            # Color based on confidence (green=high, yellow=medium, red=low)
            if confidence > 0.7:
                color = (0, 255, 0)
            elif confidence > 0.4:
                color = (255, 255, 0)
            else:
                color = (255, 0, 0)
            
            # Draw box
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{text} ({confidence:.2f})"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_draw, (x1, y1 - text_h - 4), (x1 + text_w, y1), color, -1)
            cv2.putText(img_draw, label, (x1, y1 - 2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return img_draw
