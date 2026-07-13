"""Vision processing pipeline combining multiple models."""

import cv2
import numpy as np
from typing import Dict, Optional, List, Union
from .detector import Detector
from .classifier import Classifier
from .embedding import EmbeddingModel
from .ocr import OCRModel
from .preprocessing import Preprocessor
from .config import Config


class VisionPipeline:
    """Unified pipeline for vision tasks."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.preprocessor = Preprocessor(target_size=self.config.IMAGE_SIZE)
        self.detector = None
        self.classifier = None
        self.embedding_model = None
        self.ocr_model = None
    
    def load_detector(self):
        """Lazy load detector."""
        if self.detector is None:
            print("Loading detector...")
            self.detector = Detector(self.config)
    
    def load_classifier(self):
        """Lazy load classifier."""
        if self.classifier is None:
            print("Loading classifier...")
            self.classifier = Classifier(self.config)
    
    def load_embedding_model(self):
        """Lazy load embedding model."""
        if self.embedding_model is None:
            print("Loading embedding model...")
            self.embedding_model = EmbeddingModel(self.config)
    
    def load_ocr_model(self):
        """Lazy load OCR model."""
        if self.ocr_model is None:
            print("Loading OCR model...")
            self.ocr_model = OCRModel(self.config)
    
    def process(self, image: Union[np.ndarray, str], tasks: Optional[List[str]] = None) -> Dict:
        """
        Process image through specified tasks.
        
        Args:
            image: Input image (numpy array or file path)
            tasks: List of tasks ['detect', 'classify', 'embed', 'ocr']
        
        Returns:
            Dictionary with results for each task
        """
        if tasks is None:
            tasks = ['classify']
        
        # Load image if path is provided
        if isinstance(image, str):
            image = self.preprocessor.load_image(image)
        
        results = {'original_shape': image.shape}
        
        # Detection doesn't need preprocessing
        if 'detect' in tasks:
            self.load_detector()
            results['detections'] = self.detector.detect(image)
        
        # Classification, embedding, and OCR use original image
        if 'classify' in tasks:
            self.load_classifier()
            results['classification'] = self.classifier.predict_top_k(image)
        
        if 'embed' in tasks:
            self.load_embedding_model()
            results['embedding'] = self.embedding_model.extract(image)
        
        if 'ocr' in tasks:
            self.load_ocr_model()
            results['ocr'] = self.ocr_model.extract_text_with_boxes(image)
        
        return results
    
    def process_batch(self, images: List[Union[np.ndarray, str]], tasks: Optional[List[str]] = None) -> List[Dict]:
        """
        Process batch of images.
        
        Args:
            images: List of images (numpy arrays or file paths)
            tasks: List of tasks to perform
        
        Returns:
            List of result dictionaries
        """
        results = []
        for image in images:
            result = self.process(image, tasks)
            results.append(result)
        return results
    
    def visualize_results(self, image: np.ndarray, results: Dict, save_path: Optional[str] = None) -> np.ndarray:
        """
        Visualize results on image.
        
        Args:
            image: Original image
            results: Results from process()
            save_path: Optional path to save visualization
        
        Returns:
            Image with visualizations
        """
        vis_image = image.copy()
        
        # Draw detections
        if 'detections' in results and results['detections']:
            vis_image = self.detector.draw_detections(vis_image, results['detections'])
        
        # Draw OCR results
        if 'ocr' in results and results['ocr']:
            vis_image = self.ocr_model.draw_text_boxes(vis_image, results['ocr'])
        
        # Add classification results as text
        if 'classification' in results and results['classification']:
            y_offset = 30
            for i, pred in enumerate(results['classification'][:3]):
                text = f"{pred['class']}: {pred['probability']:.3f}"
                cv2.putText(vis_image, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(vis_image, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
                y_offset += 30
        
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
        
        return vis_image
    
    def get_summary(self, results: Dict) -> str:
        """Get human-readable summary of results."""
        summary = []
        
        if 'detections' in results:
            num_detections = len(results['detections'])
            summary.append(f"Detected {num_detections} objects")
            if num_detections > 0:
                classes = [d['class'] for d in results['detections']]
                unique_classes = set(classes)
                summary.append(f"Classes: {', '.join(unique_classes)}")
        
        if 'classification' in results and results['classification']:
            top_pred = results['classification'][0]
            summary.append(f"Top prediction: {top_pred['class']} ({top_pred['probability']:.2%})")
        
        if 'embedding' in results:
            summary.append(f"Embedding dimension: {results['embedding'].shape[0]}")
        
        if 'ocr' in results:
            num_text = len(results['ocr'])
            summary.append(f"Found {num_text} text regions")
            if num_text > 0:
                texts = [r['text'] for r in results['ocr']]
                summary.append(f"Text: {' '.join(texts)}")
        
        return "\n".join(summary)
