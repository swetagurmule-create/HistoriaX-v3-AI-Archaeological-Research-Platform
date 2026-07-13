"""Complete pipeline for historical artifact analysis."""

import cv2
import numpy as np
from typing import Dict, Optional, List, Union
from .artifact_detector import ArtifactDetector
from .artifact_classifier import ArtifactClassifier
from .historical_ocr import HistoricalOCR
from .embedding import EmbeddingModel
from .preprocessing import Preprocessor
from .config import Config


class ArtifactPipeline:
    """
    Specialized pipeline for analyzing historical artifacts, manuscripts, and inscriptions.
    Fixes the issues with generic COCO/ImageNet models.
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.preprocessor = Preprocessor(target_size=self.config.IMAGE_SIZE)
        
        # Specialized models for artifacts
        self.detector = None
        self.classifier = None
        self.ocr = None
        self.embedding_model = None
        
        print("Artifact Pipeline initialized")
        print("This pipeline uses:")
        print("  - CLIP for zero-shot artifact detection (not COCO)")
        print("  - CLIP for meaningful artifact classification (not ImageNet)")
        print("  - EasyOCR with preprocessing for historical text")
    
    def load_detector(self):
        """Load artifact-specific detector."""
        if self.detector is None:
            print("\n[Loading Artifact Detector]")
            self.detector = ArtifactDetector(self.config)
    
    def load_classifier(self):
        """Load artifact-specific classifier."""
        if self.classifier is None:
            print("\n[Loading Artifact Classifier]")
            self.classifier = ArtifactClassifier(self.config)
    
    def load_ocr(self):
        """Load historical OCR."""
        if self.ocr is None:
            print("\n[Loading Historical OCR]")
            self.ocr = HistoricalOCR(self.config)
    
    def load_embedding_model(self):
        """Load embedding model."""
        if self.embedding_model is None:
            print("\n[Loading Embedding Model]")
            self.embedding_model = EmbeddingModel(self.config)
    
    def analyze_artifact(self, image: Union[np.ndarray, str], 
                        document_type: str = 'manuscript',
                        context: Optional[str] = None) -> Dict:
        """
        Complete analysis of a historical artifact.
        
        Args:
            image: Input image (numpy array or file path)
            document_type: 'manuscript', 'inscription', 'papyrus', 'artifact'
            context: Optional context (e.g., 'Egyptian', 'Roman', 'Medieval')
        
        Returns:
            Comprehensive analysis results
        """
        # Load image if path provided
        if isinstance(image, str):
            image = self.preprocessor.load_image(image)
        
        print(f"\n{'='*60}")
        print(f"ANALYZING HISTORICAL ARTIFACT")
        print(f"{'='*60}")
        print(f"Image shape: {image.shape}")
        print(f"Document type: {document_type}")
        if context:
            print(f"Context: {context}")
        
        results = {
            'image_shape': image.shape,
            'document_type': document_type,
            'context': context
        }
        
        # 1. Classify the entire artifact
        print(f"\n[1/4] Classifying artifact...")
        self.load_classifier()
        
        if context:
            classification = self.classifier.classify_with_context(image, context)
        else:
            classification = self.classifier.predict_top_k(image, k=3)
        
        results['classification'] = classification
        print(f"✓ Top prediction: {classification[0]['class']} ({classification[0]['probability']:.2%})")
        
        # 2. Detect regions of interest
        print(f"\n[2/4] Detecting artifact regions...")
        self.load_detector()
        
        # For full artifact images, classify the whole image
        full_classification = self.detector.detect_full_image(image)
        results['artifact_type'] = full_classification
        print(f"✓ Artifact type: {full_classification['class']} ({full_classification['confidence']:.2%})")
        
        # Also detect sub-regions
        detections = self.detector.detect(image)
        results['detections'] = detections
        print(f"✓ Found {len(detections)} regions")
        
        # 3. Extract text (if document contains text)
        if document_type in ['manuscript', 'inscription', 'papyrus', 'document']:
            print(f"\n[3/4] Extracting text...")
            self.load_ocr()
            
            ocr_results = self.ocr.extract_with_confidence_analysis(image, document_type)
            results['ocr'] = ocr_results
            
            if ocr_results['text']:
                print(f"✓ Extracted text: \"{ocr_results['text'][:100]}...\"")
                print(f"✓ Word count: {ocr_results['word_count']}")
                print(f"✓ Avg confidence: {ocr_results['avg_confidence']:.2%}")
            else:
                print(f"⚠️  No text detected")
        else:
            print(f"\n[3/4] Skipping OCR (not a text document)")
            results['ocr'] = None
        
        # 4. Extract embedding for similarity search
        print(f"\n[4/4] Extracting features...")
        self.load_embedding_model()
        
        embedding = self.embedding_model.extract(image)
        results['embedding'] = embedding
        print(f"✓ Embedding extracted: {embedding.shape[0]} dimensions")
        
        return results
    
    def get_summary(self, results: Dict) -> str:
        """Generate human-readable summary."""
        lines = []
        lines.append("="*60)
        lines.append("ARTIFACT ANALYSIS SUMMARY")
        lines.append("="*60)
        
        # Classification
        if 'classification' in results:
            top = results['classification'][0]
            lines.append(f"\nArtifact Type: {top['class'].replace('_', ' ').title()}")
            lines.append(f"Confidence: {top['probability']:.1%}")
            lines.append(f"Description: {top['description']}")
        
        # Detections
        if 'detections' in results and results['detections']:
            lines.append(f"\nDetected Regions: {len(results['detections'])}")
            for i, det in enumerate(results['detections'][:3], 1):
                lines.append(f"  {i}. {det['class']} ({det['confidence']:.1%})")
        
        # OCR
        if results.get('ocr'):
            ocr = results['ocr']
            if ocr['text']:
                lines.append(f"\nExtracted Text:")
                lines.append(f"  \"{ocr['text'][:200]}...\"" if len(ocr['text']) > 200 else f"  \"{ocr['text']}\"")
                lines.append(f"  Words: {ocr['word_count']}, Confidence: {ocr['avg_confidence']:.1%}")
                lines.append(f"  Readable: {'Yes' if ocr['readable'] else 'No'}")
            else:
                lines.append(f"\nNo text detected in image")
        
        # Embedding
        if 'embedding' in results:
            lines.append(f"\nFeature Vector: {results['embedding'].shape[0]} dimensions")
            lines.append(f"  (Can be used for similarity search)")
        
        return "\n".join(lines)
    
    def visualize_results(self, image: np.ndarray, results: Dict, 
                         save_path: Optional[str] = None) -> np.ndarray:
        """Create visualization of analysis results."""
        vis_image = image.copy()
        
        # Draw detections
        if 'detections' in results and results['detections']:
            for det in results['detections']:
                x1, y1, x2, y2 = map(int, det['bbox'])
                label = f"{det['class']}: {det['confidence']:.2f}"
                
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis_image, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw OCR results
        if results.get('ocr') and results['ocr'].get('details'):
            self.load_ocr()
            vis_image = self.ocr.draw_text_boxes(vis_image, results['ocr']['details'])
        
        # Add classification label at top
        if 'classification' in results:
            top = results['classification'][0]
            label = f"Type: {top['class']} ({top['probability']:.1%})"
            
            cv2.rectangle(vis_image, (10, 10), (500, 50), (0, 0, 0), -1)
            cv2.putText(vis_image, label, (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        if save_path:
            cv2.imwrite(save_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
            print(f"\n✓ Visualization saved to: {save_path}")
        
        return vis_image
    
    def compare_artifacts(self, image1: np.ndarray, image2: np.ndarray) -> Dict:
        """Compare two artifacts for similarity."""
        self.load_embedding_model()
        
        emb1 = self.embedding_model.extract(image1)
        emb2 = self.embedding_model.extract(image2)
        
        similarity = self.embedding_model.compute_similarity(emb1, emb2)
        
        return {
            'similarity': similarity,
            'similar': similarity > 0.7,
            'interpretation': self._interpret_similarity(similarity)
        }
    
    def _interpret_similarity(self, similarity: float) -> str:
        """Interpret similarity score."""
        if similarity > 0.9:
            return "Nearly identical artifacts"
        elif similarity > 0.7:
            return "Very similar artifacts (same type/period)"
        elif similarity > 0.5:
            return "Moderately similar (related category)"
        elif similarity > 0.3:
            return "Somewhat similar"
        else:
            return "Different artifacts"
