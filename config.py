"""Configuration settings for vision models."""

import torch


class Config:
    """Central configuration for all vision models."""
    
    # Model paths
    DETECTOR_MODEL_PATH = "models/detector.pth"
    CLASSIFIER_MODEL_PATH = "models/classifier.pth"
    EMBEDDING_MODEL_PATH = "models/embedding.pth"
    OCR_MODEL_PATH = "models/ocr.pth"
    
    # Image settings
    IMAGE_SIZE = (224, 224)
    DETECTOR_IMAGE_SIZE = (640, 640)
    BATCH_SIZE = 32
    
    # Detection settings
    DETECTION_THRESHOLD = 0.5
    NMS_THRESHOLD = 0.4
    MAX_DETECTIONS = 100
    
    # Classification settings
    NUM_CLASSES = 1000
    TOP_K = 5
    
    # Embedding settings
    EMBEDDING_DIM = 512
    
    # OCR settings
    OCR_LANGUAGES = ['en']
    OCR_CONFIDENCE_THRESHOLD = 0.5
    
    # Preprocessing settings
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    
    # Device settings
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model backends
    DETECTOR_BACKEND = "yolov8"  # yolov8, fasterrcnn
    CLASSIFIER_BACKEND = "resnet50"  # resnet50, efficientnet, vit
    EMBEDDING_BACKEND = "resnet50"  # resnet50, clip
    OCR_BACKEND = "easyocr"  # easyocr, tesseract, paddleocr
    
    @classmethod
    def from_dict(cls, config_dict):
        """Load config from dictionary."""
        config = cls()
        for key, value in config_dict.items():
            if hasattr(config, key):
                setattr(config, key, value)
        return config
