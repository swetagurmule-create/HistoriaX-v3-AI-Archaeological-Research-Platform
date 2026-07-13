"""Optical Character Recognition model."""

import cv2
import numpy as np
from typing import List, Dict, Optional
from .config import Config


class OCRModel:
    """OCR model for text detection and recognition."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.reader = None
        self.backend = self.config.OCR_BACKEND
        self.load_model()
    
    def load_model(self):
        """Load OCR model."""
        try:
            if self.backend == "easyocr":
                import easyocr
                self.reader = easyocr.Reader(
                    self.config.OCR_LANGUAGES,
                    gpu=self.config.DEVICE == "cuda"
                )
            elif self.backend == "tesseract":
                try:
                    import pytesseract
                    self.reader = pytesseract
                except ImportError:
                    print("pytesseract not installed, falling back to EasyOCR")
                    self._load_easyocr()
            elif self.backend == "paddleocr":
                try:
                    from paddleocr import PaddleOCR
                    self.reader = PaddleOCR(
                        use_angle_cls=True,
                        lang='en',
                        use_gpu=self.config.DEVICE == "cuda"
                    )
                except ImportError:
                    print("PaddleOCR not installed, falling back to EasyOCR")
                    self._load_easyocr()
            else:
                self._load_easyocr()
        except Exception as e:
            print(f"Error loading OCR model: {e}")
            print("OCR functionality will be limited")
    
    def _load_easyocr(self):
        """Fallback to EasyOCR."""
        try:
            import easyocr
            self.reader = easyocr.Reader(
                self.config.OCR_LANGUAGES,
                gpu=self.config.DEVICE == "cuda"
            )
            self.backend = "easyocr"
        except ImportError:
            print("EasyOCR not installed. Install with: pip install easyocr")
            self.reader = None
    
    def detect_text(self, image: np.ndarray) -> List[Dict]:
        """
        Detect text regions in image.
        
        Returns:
            List of text regions: [{'bbox': [x1, y1, x2, y2], 'confidence': float}]
        """
        if self.reader is None:
            return []
        
        if self.backend == "easyocr":
            results = self.reader.detect(image)
            text_regions = []
            
            if results and len(results) > 0:
                boxes, scores = results[0], results[1]
                for box, score in zip(boxes, scores):
                    x1, y1 = box[0]
                    x2, y2 = box[2]
                    text_regions.append({
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(score)
                    })
            
            return text_regions
        
        return []
    
    def recognize_text(self, image: np.ndarray) -> str:
        """
        Recognize text from image.
        
        Returns:
            Extracted text string
        """
        if self.reader is None:
            return ""
        
        if self.backend == "easyocr":
            results = self.reader.readtext(image, detail=0)
            return " ".join(results)
        elif self.backend == "tesseract":
            return self.reader.image_to_string(image)
        elif self.backend == "paddleocr":
            results = self.reader.ocr(image, cls=True)
            if results and results[0]:
                return " ".join([line[1][0] for line in results[0]])
        
        return ""
    
    def extract_text_with_boxes(self, image: np.ndarray, min_confidence: Optional[float] = None) -> List[Dict]:
        """
        Detect and recognize text with bounding boxes.
        
        Returns:
            List of: [{'text': str, 'bbox': [x1, y1, x2, y2], 'confidence': float}]
        """
        if self.reader is None:
            return []
        
        if min_confidence is None:
            min_confidence = self.config.OCR_CONFIDENCE_THRESHOLD
        
        results = []
        
        if self.backend == "easyocr":
            ocr_results = self.reader.readtext(image)
            
            for detection in ocr_results:
                bbox, text, confidence = detection
                
                if confidence >= min_confidence:
                    # Convert bbox to [x1, y1, x2, y2] format
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    x1, y1 = min(x_coords), min(y_coords)
                    x2, y2 = max(x_coords), max(y_coords)
                    
                    results.append({
                        'text': text,
                        'bbox': [float(x1), float(y1), float(x2), float(y2)],
                        'confidence': float(confidence)
                    })
        
        elif self.backend == "tesseract":
            import pytesseract
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            for i in range(len(data['text'])):
                if int(data['conf'][i]) >= min_confidence * 100:
                    text = data['text'][i].strip()
                    if text:
                        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                        results.append({
                            'text': text,
                            'bbox': [float(x), float(y), float(x + w), float(y + h)],
                            'confidence': float(data['conf'][i]) / 100.0
                        })
        
        elif self.backend == "paddleocr":
            ocr_results = self.reader.ocr(image, cls=True)
            
            if ocr_results and ocr_results[0]:
                for line in ocr_results[0]:
                    bbox, (text, confidence) = line
                    
                    if confidence >= min_confidence:
                        x_coords = [point[0] for point in bbox]
                        y_coords = [point[1] for point in bbox]
                        x1, y1 = min(x_coords), min(y_coords)
                        x2, y2 = max(x_coords), max(y_coords)
                        
                        results.append({
                            'text': text,
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': float(confidence)
                        })
        
        return results
    
    def draw_text_boxes(self, image: np.ndarray, text_results: List[Dict]) -> np.ndarray:
        """Draw text bounding boxes on image."""
        img_draw = image.copy()
        
        for result in text_results:
            x1, y1, x2, y2 = map(int, result['bbox'])
            text = result['text']
            confidence = result['confidence']
            
            # Draw box
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{text} ({confidence:.2f})"
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_draw, (x1, y1 - text_h - 4), (x1 + text_w, y1), (0, 255, 0), -1)
            cv2.putText(img_draw, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return img_draw
