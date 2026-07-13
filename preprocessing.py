"""Image preprocessing utilities."""

import cv2
import numpy as np
import torch
from typing import Tuple, Optional, Union
from PIL import Image


class Preprocessor:
    """Handles image preprocessing for vision models."""
    
    def __init__(self, target_size: Tuple[int, int] = (224, 224)):
        self.target_size = target_size
    
    def load_image(self, image_path: str) -> np.ndarray:
        """Load image from file path."""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image from {image_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def resize(self, image: np.ndarray, size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Resize image to target size."""
        if size is None:
            size = self.target_size
        return cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    
    def resize_keep_aspect(self, image: np.ndarray, size: Tuple[int, int]) -> Tuple[np.ndarray, float]:
        """Resize image keeping aspect ratio with padding."""
        h, w = image.shape[:2]
        target_h, target_w = size
        
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        padded[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized
        
        return padded, scale
    
    def normalize(self, image: np.ndarray, mean: Optional[list] = None, std: Optional[list] = None) -> np.ndarray:
        """Normalize image with mean and std."""
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]
        
        image = image.astype(np.float32) / 255.0
        for i in range(3):
            image[:, :, i] = (image[:, :, i] - mean[i]) / std[i]
        return image
    
    def to_tensor(self, image: np.ndarray) -> torch.Tensor:
        """Convert numpy array to PyTorch tensor."""
        if len(image.shape) == 3:
            # HWC to CHW
            image = np.transpose(image, (2, 0, 1))
        return torch.from_numpy(image).float()
    
    def preprocess(self, image: np.ndarray, normalize: bool = True) -> np.ndarray:
        """Full preprocessing pipeline."""
        image = self.resize(image)
        if normalize:
            image = self.normalize(image)
        return image
    
    def preprocess_batch(self, images: list) -> torch.Tensor:
        """Preprocess batch of images."""
        processed = [self.preprocess(img) for img in images]
        tensors = [self.to_tensor(img) for img in processed]
        return torch.stack(tensors)
    
    def denormalize(self, image: np.ndarray, mean: Optional[list] = None, std: Optional[list] = None) -> np.ndarray:
        """Denormalize image for visualization."""
        if mean is None:
            mean = [0.485, 0.456, 0.406]
        if std is None:
            std = [0.229, 0.224, 0.225]
        
        image = image.copy()
        for i in range(3):
            image[:, :, i] = image[:, :, i] * std[i] + mean[i]
        image = (image * 255.0).clip(0, 255).astype(np.uint8)
        return image
