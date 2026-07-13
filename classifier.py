"""Image classification model."""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from typing import Dict, List, Optional
from .config import Config
import json


class Classifier:
    """Image classification model."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.model = None
        self.device = torch.device(self.config.DEVICE)
        self.class_names = []
        self.transform = self._create_transform()
        self.load_model()
    
    def _create_transform(self):
        """Create image transformation pipeline."""
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.NORMALIZE_MEAN,
                std=self.config.NORMALIZE_STD
            )
        ])
    
    def load_model(self):
        """Load classification model."""
        try:
            if self.config.CLASSIFIER_BACKEND == "resnet50":
                self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            elif self.config.CLASSIFIER_BACKEND == "efficientnet":
                self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            elif self.config.CLASSIFIER_BACKEND == "vit":
                self.model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
            else:
                self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
            
            self.model.to(self.device)
            self.model.eval()
            
            # Load ImageNet class names
            self.class_names = self._load_imagenet_classes()
            
        except Exception as e:
            print(f"Error loading classifier: {e}")
            raise
    
    def _load_imagenet_classes(self) -> List[str]:
        """Load ImageNet class names."""
        # Simplified ImageNet classes (first 20 for demo)
        classes = [
            'tench', 'goldfish', 'great white shark', 'tiger shark', 'hammerhead',
            'electric ray', 'stingray', 'cock', 'hen', 'ostrich', 'brambling',
            'goldfinch', 'house finch', 'junco', 'indigo bunting', 'robin',
            'bulbul', 'jay', 'magpie', 'chickadee'
        ]
        # In production, load full 1000 classes from a file
        return classes + [f'class_{i}' for i in range(20, 1000)]
    
    def predict(self, image: np.ndarray) -> Dict[str, float]:
        """
        Classify image.
        
        Returns:
            Dictionary of class probabilities: {'class_name': probability}
        """
        # Preprocess image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # Convert to dictionary
        probs = probabilities.cpu().numpy()
        predictions = {
            self.class_names[i]: float(probs[i])
            for i in range(len(self.class_names))
        }
        
        return predictions
    
    def predict_top_k(self, image: np.ndarray, k: Optional[int] = None) -> List[Dict]:
        """
        Get top-k predictions.
        
        Returns:
            List of dicts: [{'class': str, 'probability': float, 'rank': int}]
        """
        if k is None:
            k = self.config.TOP_K
        
        # Preprocess image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # Get top-k
        top_probs, top_indices = torch.topk(probabilities, k)
        
        results = []
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
            results.append({
                'rank': i + 1,
                'class': self.class_names[idx.item()],
                'class_id': idx.item(),
                'probability': prob.item()
            })
        
        return results
    
    def predict_batch(self, images: List[np.ndarray], k: Optional[int] = None) -> List[List[Dict]]:
        """Predict on batch of images."""
        if k is None:
            k = self.config.TOP_K
        
        # Preprocess batch
        img_tensors = torch.stack([self.transform(img) for img in images]).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(img_tensors)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # Get top-k for each image
        batch_results = []
        for probs in probabilities:
            top_probs, top_indices = torch.topk(probs, k)
            results = []
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                results.append({
                    'rank': i + 1,
                    'class': self.class_names[idx.item()],
                    'class_id': idx.item(),
                    'probability': prob.item()
                })
            batch_results.append(results)
        
        return batch_results
