"""Vision model package for detection, classification, OCR, and embeddings."""

from .detector import Detector
from .classifier import Classifier
from .embedding import EmbeddingModel
from .ocr import OCRModel
from .preprocessing import Preprocessor
from .pipeline import VisionPipeline
from .config import Config

# Specialized modules for historical artifacts
from .artifact_detector import ArtifactDetector
from .artifact_classifier import ArtifactClassifier
from .historical_ocr import HistoricalOCR
from .artifact_pipeline import ArtifactPipeline

__all__ = [
    # Generic vision models
    'Detector',
    'Classifier',
    'EmbeddingModel',
    'OCRModel',
    'Preprocessor',
    'VisionPipeline',
    'Config',
    # Artifact-specific models
    'ArtifactDetector',
    'ArtifactClassifier',
    'HistoricalOCR',
    'ArtifactPipeline'
]
