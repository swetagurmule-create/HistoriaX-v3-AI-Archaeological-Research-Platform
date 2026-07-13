"""Specialized classifier for historical artifacts using CLIP zero-shot classification."""

import torch
import numpy as np
from typing import List, Dict, Optional
from PIL import Image
from .config import Config


class ArtifactClassifier:
    """
    Classifier for historical artifacts using CLIP.
    No retraining needed - uses text prompts for classification.
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.device = torch.device(self.config.DEVICE)
        self.model = None
        self.processor = None
        
        # Artifact-specific categories with detailed descriptions
        self.categories = {
            'manuscript': 'a historical manuscript with handwritten text',
            'ancient_document': 'an ancient document or scroll with text',
            'inscription': 'a stone inscription or carved text',
            'papyrus': 'an ancient papyrus document',
            'tablet': 'a clay or stone tablet with writing',
            'pottery': 'ancient pottery or ceramic vessel',
            'sculpture': 'a historical sculpture or statue',
            'coin': 'an ancient coin or currency',
            'jewelry': 'historical jewelry or ornament',
            'painting': 'a historical painting or artwork',
            'fresco': 'a wall fresco or mural',
            'hieroglyphics': 'Egyptian hieroglyphic writing',
            'cuneiform': 'cuneiform script on clay tablet',
            'seal': 'an ancient seal or stamp',
            'artifact': 'a general historical artifact',
            'relic': 'a religious or historical relic',
            'archaeological_find': 'an archaeological discovery',
            'book': 'an ancient or medieval book',
            'map': 'a historical map or chart',
            'textile': 'historical textile or fabric'
        }
        
        self.load_model()
    
    def load_model(self):
        """Load CLIP model for zero-shot classification."""
        try:
            from transformers import CLIPProcessor, CLIPModel
            
            print("Loading CLIP model for artifact classification...")
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✓ CLIP classifier loaded with {len(self.categories)} artifact categories")
            
        except ImportError:
            print("⚠️  transformers not installed")
            print("   Install with: pip install transformers")
            raise
    
    def predict(self, image: np.ndarray, custom_categories: Optional[Dict[str, str]] = None) -> Dict[str, float]:
        """
        Classify artifact image.
        
        Args:
            image: Input image (numpy array)
            custom_categories: Optional custom categories dict {name: description}
        
        Returns:
            Dictionary of category probabilities
        """
        categories = custom_categories or self.categories
        
        # Convert to PIL
        pil_image = Image.fromarray(image)
        
        # Prepare text prompts
        text_prompts = [desc for desc in categories.values()]
        
        # Process inputs
        inputs = self.processor(
            text=text_prompts,
            images=pil_image,
            return_tensors="pt",
            padding=True
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)[0]
        
        # Create results dictionary
        category_names = list(categories.keys())
        results = {
            category_names[i]: float(probs[i].item())
            for i in range(len(category_names))
        }
        
        return results
    
    def predict_top_k(self, image: np.ndarray, k: int = 5, 
                     custom_categories: Optional[Dict[str, str]] = None) -> List[Dict]:
        """
        Get top-k artifact predictions.
        
        Returns:
            List of dicts: [{'rank': int, 'class': str, 'probability': float, 'description': str}]
        """
        categories = custom_categories or self.categories
        predictions = self.predict(image, custom_categories)
        
        # Sort by probability
        sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:k]
        
        results = []
        for i, (category, prob) in enumerate(sorted_preds, 1):
            results.append({
                'rank': i,
                'class': category,
                'probability': prob,
                'description': categories[category]
            })
        
        return results
    
    def classify_with_context(self, image: np.ndarray, context: str) -> List[Dict]:
        """
        Classify with additional context.
        
        Args:
            image: Input image
            context: Context string (e.g., "Egyptian", "Roman", "Medieval")
        
        Returns:
            Top predictions with context-aware categories
        """
        # Create context-specific categories
        context_categories = {
            f"{context.lower()}_{key}": f"{context} {desc}"
            for key, desc in self.categories.items()
        }
        
        return self.predict_top_k(image, k=5, custom_categories=context_categories)
    
    def add_custom_categories(self, new_categories: Dict[str, str]):
        """
        Add custom artifact categories.
        
        Args:
            new_categories: Dict of {category_name: description}
        """
        self.categories.update(new_categories)
        print(f"✓ Added {len(new_categories)} custom categories")
        print(f"  Total categories: {len(self.categories)}")
