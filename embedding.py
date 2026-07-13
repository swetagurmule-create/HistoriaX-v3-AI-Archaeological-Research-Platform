"""Image embedding model for feature extraction."""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
from typing import Optional, List
from .config import Config


class EmbeddingModel:
    """Extract feature embeddings from images."""
    
    def __init__(self, config: Config = None, embedding_dim: Optional[int] = None):
        self.config = config or Config()
        self.embedding_dim = embedding_dim or self.config.EMBEDDING_DIM
        self.model = None
        self.device = torch.device(self.config.DEVICE)
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
        """Load embedding model."""
        try:
            if self.config.EMBEDDING_BACKEND == "resnet50":
                # Load ResNet50 and remove final classification layer
                base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
                self.model = nn.Sequential(*list(base_model.children())[:-1])
                self.embedding_dim = 2048
            elif self.config.EMBEDDING_BACKEND == "resnet18":
                base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
                self.model = nn.Sequential(*list(base_model.children())[:-1])
                self.embedding_dim = 512
            elif self.config.EMBEDDING_BACKEND == "efficientnet":
                base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
                self.model = nn.Sequential(*list(base_model.children())[:-1])
                self.embedding_dim = 1280
            else:
                # Default to ResNet50
                base_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
                self.model = nn.Sequential(*list(base_model.children())[:-1])
                self.embedding_dim = 2048
            
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            raise
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        Extract embedding vector from image.
        
        Returns:
            Embedding vector of shape (embedding_dim,)
        """
        # Preprocess image
        img_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            embedding = self.model(img_tensor)
            embedding = embedding.squeeze().cpu().numpy()
        
        # Normalize embedding
        embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
        
        return embedding
    
    def extract_batch(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Extract embeddings for batch of images.
        
        Returns:
            Array of shape (batch_size, embedding_dim)
        """
        # Preprocess batch
        img_tensors = torch.stack([self.transform(img) for img in images]).to(self.device)
        
        # Extract features
        with torch.no_grad():
            embeddings = self.model(img_tensors)
            embeddings = embeddings.squeeze().cpu().numpy()
        
        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        embeddings = embeddings / norms
        
        return embeddings
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        return float(dot_product / (norm1 * norm2 + 1e-8))
    
    def compute_similarity_matrix(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """
        Compute pairwise similarity matrix between two sets of embeddings.
        
        Args:
            embeddings1: Array of shape (n, embedding_dim)
            embeddings2: Array of shape (m, embedding_dim)
        
        Returns:
            Similarity matrix of shape (n, m)
        """
        # Normalize embeddings
        embeddings1 = embeddings1 / (np.linalg.norm(embeddings1, axis=1, keepdims=True) + 1e-8)
        embeddings2 = embeddings2 / (np.linalg.norm(embeddings2, axis=1, keepdims=True) + 1e-8)
        
        # Compute cosine similarity
        similarity_matrix = np.dot(embeddings1, embeddings2.T)
        return similarity_matrix
    
    def find_similar(self, query_embedding: np.ndarray, database_embeddings: np.ndarray, top_k: int = 5) -> List[dict]:
        """
        Find most similar embeddings in database.
        
        Returns:
            List of dicts with 'index' and 'similarity' keys
        """
        similarities = np.dot(database_embeddings, query_embedding)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append({
                'index': int(idx),
                'similarity': float(similarities[idx])
            })
        
        return results
