"""Object detection model."""

import cv2
import numpy as np
import torch
import torchvision
from typing import List, Dict, Optional
from .config import Config


class Detector:
    """Object detection model for identifying and localizing objects."""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.model = None
        self.device = torch.device(self.config.DEVICE)
        self.class_names = self._load_coco_names()
        self.load_model()
    
    def _load_coco_names(self) -> List[str]:
        """Load COCO class names."""
        return [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
    
    def load_model(self):
        """Load detection model."""
        try:
            if self.config.DETECTOR_BACKEND == "yolov8":
                # Try to use ultralytics YOLOv8
                try:
                    from ultralytics import YOLO
                    self.model = YOLO('yolov8n.pt')  # nano model
                    self.backend = "yolov8"
                except ImportError:
                    print("Ultralytics not installed, falling back to Faster R-CNN")
                    self._load_fasterrcnn()
            else:
                self._load_fasterrcnn()
        except Exception as e:
            print(f"Error loading detector: {e}")
            self._load_fasterrcnn()
    
    def _load_fasterrcnn(self):
        """Load Faster R-CNN from torchvision."""
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        )
        self.model.to(self.device)
        self.model.eval()
        self.backend = "fasterrcnn"
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect objects in image.
        
        Returns:
            List of detections with format:
            [{'bbox': [x1, y1, x2, y2], 'class': str, 'confidence': float}]
        """
        if self.backend == "yolov8":
            return self._detect_yolo(image)
        else:
            return self._detect_fasterrcnn(image)
    
    def _detect_yolo(self, image: np.ndarray) -> List[Dict]:
        """Detect using YOLOv8."""
        results = self.model(image, conf=self.config.DETECTION_THRESHOLD)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()
                conf = float(boxes.conf[i])
                cls_id = int(boxes.cls[i])
                
                if cls_id < len(self.class_names):
                    detections.append({
                        'bbox': box.tolist(),
                        'class': self.class_names[cls_id],
                        'confidence': conf,
                        'class_id': cls_id
                    })
        
        return detections
    
    def _detect_fasterrcnn(self, image: np.ndarray) -> List[Dict]:
        """Detect using Faster R-CNN."""
        # Convert to tensor
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            predictions = self.model(img_tensor)[0]
        
        detections = []
        boxes = predictions['boxes'].cpu().numpy()
        scores = predictions['scores'].cpu().numpy()
        labels = predictions['labels'].cpu().numpy()
        
        for i in range(len(boxes)):
            if scores[i] >= self.config.DETECTION_THRESHOLD:
                cls_id = int(labels[i]) - 1  # COCO labels are 1-indexed
                if 0 <= cls_id < len(self.class_names):
                    detections.append({
                        'bbox': boxes[i].tolist(),
                        'class': self.class_names[cls_id],
                        'confidence': float(scores[i]),
                        'class_id': cls_id
                    })
        
        return detections
    
    def non_max_suppression(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: Optional[float] = None) -> np.ndarray:
        """Apply NMS to remove overlapping boxes."""
        if iou_threshold is None:
            iou_threshold = self.config.NMS_THRESHOLD
        
        boxes_tensor = torch.from_numpy(boxes).float()
        scores_tensor = torch.from_numpy(scores).float()
        
        keep_indices = torchvision.ops.nms(boxes_tensor, scores_tensor, iou_threshold)
        return keep_indices.numpy()
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """Draw bounding boxes on image."""
        img_draw = image.copy()
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            label = f"{det['class']}: {det['confidence']:.2f}"
            
            # Draw box
            cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label background
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(img_draw, (x1, y1 - text_h - 4), (x1 + text_w, y1), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(img_draw, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        return img_draw
