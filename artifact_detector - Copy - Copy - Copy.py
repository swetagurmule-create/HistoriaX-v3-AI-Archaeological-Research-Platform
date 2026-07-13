"""Specialized detector for historical artifacts and manuscripts."""

import cv2
import numpy as np
import torch
from typing import List, Dict, Optional
from .config import Config


class ArtifactDetector:
    """
    Detector specialized for historical artifacts, manuscripts, and documents.
    Uses CLIP for zero-shot detection instead of COCO-trained models.
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.device = torch.device(self.config.DEVICE)
        self.model = None
        self.processor = None
        
        # Artifact-specific categories
        self.artifact_classes = [
            'manuscript', 'ancient manuscript', 'historical document',
            'inscription', 'stone tablet', 'papyrus',
            'artifact', 'pottery', 'sculpture', 'statue',
            'coin', 'medal', 'jewelry',
            'painting', 'fresco', 'mural',
            'hieroglyphics', 'cuneiform', 'ancient text',
            'seal', 'stamp', 'emblem',
            'archaeological find', 'relic', 'antiquity'
        ]
        
        self.load_model()
    
    def load_model(self):
        """Load CLIP model for zero-shot artifact detection."""
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            print("Loading CLIP model for artifact detection...")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✓ CLIP model loaded with {len(self.artifact_classes)} artifact categories")
            
        except ImportError:
            print("⚠️  transformers not installed. Install with: pip install transformers")
            print("   Falling back to region proposal method")
            self.model = None
    
    def detect_regions(self, image: np.ndarray) -> List[Dict]:
        """
        Detect potential artifact regions using traditional CV methods.
        Useful when CLIP is not available or as preprocessing.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        regions = []
        h, w = image.shape[:2]
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by area (adjust thresholds based on your images)
            if area < (h * w * 0.01) or area > (h * w * 0.95):
                continue
            
            x, y, w_box, h_box = cv2.boundingRect(contour)
            
            regions.append({
                'bbox': [float(x), float(y), float(x + w_box), float(y + h_box)],
                'area': float(area),
                'aspect_ratio': float(w_box / h_box) if h_box > 0 else 0
            })
        
        return regions
    
    def classify_region(self, image: np.ndarray, bbox: List[float]) -> Dict:
        """Classify a region using CLIP zero-shot classification."""
        if self.model is None:
            return {'class': 'artifact', 'confidence': 0.5}
        
        # Crop region
        x1, y1, x2, y2 = map(int, bbox)
        region = image[y1:y2, x1:x2]
        
        if region.size == 0:
            return {'class': 'unknown', 'confidence': 0.0}
        
        # Prepare inputs
        from PIL import Image
        pil_image = Image.fromarray(region)
        
        inputs = self.processor(
            text=self.artifact_classes,
            images=pil_image,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        # Get top prediction
        top_prob, top_idx = probs[0].max(dim=0)
        
        return {
            'class': self.artifact_classes[top_idx.item()],
            'confidence': float(top_prob.item())
        }
    
    def detect(self, image: np.ndarray, confidence_threshold: Optional[float] = None) -> List[Dict]:
        """
        Detect artifacts in image.
        
        Returns:
            List of detections with format:
            [{'bbox': [x1, y1, x2, y2], 'class': str, 'confidence': float}]
        """
        if confidence_threshold is None:
            confidence_threshold = self.config.DETECTION_THRESHOLD
        
        # Step 1: Find potential regions
        regions = self.detect_regions(image)
        
        # Step 2: Classify each region with CLIP
        detections = []
        
        for region in regions:
            classification = self.classify_region(image, region['bbox'])
            
            if classification['confidence'] >= confidence_threshold:
                detections.append({
                    'bbox': region['bbox'],
                    'class': classification['class'],
                    'confidence': classification['confidence'],
                    'area': region['area']
                })
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        return detections
    
    def detect_full_image(self, image: np.ndarray) -> Dict:
        """
        Classify the entire image as an artifact type.
        Useful when the whole image is one artifact.
        """
        if self.model is None:
            return {'class': 'artifact', 'confidence': 0.5}
        
        from PIL import Image
        pil_image = Image.fromarray(image)
        
        inputs = self.processor(
            text=self.artifact_classes,
            images=pil_image,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)
        
        # Get top 3 predictions
        top_probs, top_indices = probs[0].topk(3)
        
        results = []
        for prob, idx in zip(top_probs, top_indices):
            results.append({
                'class': self.artifact_classes[idx.item()],
                'confidence': float(prob.item())
            })
        
        return results[0]  # Return top prediction
    
    def add_custom_classes(self, classes: List[str]):
        """Add custom artifact classes for your specific use case."""
        self.artifact_classes.extend(classes)
        print(f"✓ Added {len(classes)} custom classes. Total: {len(self.artifact_classes)}")
