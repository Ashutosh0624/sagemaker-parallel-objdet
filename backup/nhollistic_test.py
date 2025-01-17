import os
import cv2
import json
import time
import numpy as np
import logging
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import tensorflow as tf
from collections import defaultdict
from torch.nn import Module, Sequential, Conv2d, ReLU, Sigmoid
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import scipy.stats as stats
import torchvision.transforms as T
import torchvision.transforms.functional as F
import random
import yaml
import argparse



class VOCDetectionDataset(Dataset):
    def __init__(self, data_dir: str, transforms=None, train=True):
        """
        Enhanced Pascal VOC Detection Dataset with Robust Image Loading

        Args:
            data_dir (str): Path to the dataset directory
            transforms (callable, optional): Optional transform to be applied on a sample
            train (bool): Whether in training or validation mode
        """
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Convert to Path object for easier manipulation
        self.data_dir = Path(data_dir)
        self.train = train

        # Explicitly set images directory
        self.images_dir = self.data_dir

        # Load annotations
        self.annotations = self._load_annotations()

        # Class mapping (keep as is)
        self.class_to_idx = {
            'background': 0,
            'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4, 'bottle': 5,
            'bus': 6, 'car': 7, 'cat': 8, 'chair': 9, 'cow': 10,
            'diningtable': 11, 'dog': 12, 'horse': 13, 'motorbike': 14,
            'person': 15, 'pottedplant': 16, 'sheep': 17, 'sofa': 18,
            'train': 19, 'tvmonitor': 20
        }

        # Prepare image ids
        self.image_ids = list(self.annotations.keys())

        # Store transforms
        self.transforms = transforms or self.get_transform(train)

    def _robust_image_load(self, img_path):
        """
        Robust method to load images with multiple strategies
        """
        strategies = [
            # OpenCV with color conversion
            lambda p: cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) if cv2.imread(str(p)) is not None else None,

            # PIL image loading
            lambda p: np.array(Image.open(p).convert('RGB')),

            # Numpy loading
            lambda p: np.array(Image.open(p))
        ]

        for strategy in strategies:
            try:
                image = strategy(img_path)
                if image is not None and image.size > 0:
                    return image
            except Exception as e:
                self.logger.warning(f"Image load attempt failed for {img_path}: {e}")

        raise ValueError(f"Could not load image: {img_path}")

    def _load_annotations(self):
        """
        Load annotations with multiple fallback methods
        """
        # If training, use training annotations
        if self.train:
            annotation_paths = [
                self.data_dir.parent / 'train' / "ground_truth.json",
                self.data_dir.parent / "ground_truth.json",
                self.data_dir / "ground_truth.json"
            ]
        else:
            annotation_paths = [
                self.data_dir.parent / 'valid' / "ground_truth.json",
                self.data_dir.parent / "ground_truth.json",
                self.data_dir / "ground_truth.json"
            ]

        print("\n--- Attempting to load annotations ---")
        print(f"Current data directory: {self.data_dir}")
        print(f"Parent directory: {self.data_dir.parent}")
        print("Checking these annotation paths:")
        for path in annotation_paths:
            print(f"  - {path}")
            print(f"    Exists: {path.exists()}")

        for ann_path in annotation_paths:
            try:
                if ann_path.exists():
                    with open(ann_path, 'r') as f:
                        annotations = json.load(f)

                    self.logger.info(f"Loaded annotations from {ann_path}")

                    # Detailed annotation validation
                    print("\n--- Annotation Details ---")
                    print(f"Total number of entries: {len(annotations)}")

                    # Print first few annotation keys
                    sample_keys = list(annotations.keys())[:5]
                    print("Sample annotation keys:")
                    for key in sample_keys:
                        print(f"  {key}: {len(annotations[key])} annotations")

                    # Diagnostic information about annotation IDs
                    print("\n--- Annotation ID Characteristics ---")

                    # Unique patterns in annotation IDs
                    unique_patterns = set()
                    hash_patterns = set()
                    for img_id in annotations.keys():
                        # Extract unique prefixes
                        prefix = '_'.join(img_id.split('_')[:2])
                        unique_patterns.add(prefix)

                        # Extract hash patterns
                        if '.rf.' in img_id:
                            hash_pattern = img_id.split('.rf.')[1].split('.')[0]
                            hash_patterns.add(hash_pattern)

                    print("Unique ID Prefixes:")
                    for pattern in sorted(unique_patterns)[:10]:
                        print(f"  - {pattern}")

                    print("\nSample Hash Patterns (first 10):")
                    for pattern in list(sorted(hash_patterns))[:10]:
                        print(f"  - {pattern}")

                    # Basic validation
                    if not annotations:
                        self.logger.warning("Annotations file is empty")
                        continue

                    return annotations
            except Exception as e:
                self.logger.warning(f"Failed to load annotations from {ann_path}: {e}")
                # Print more details about the error
                import traceback
                traceback.print_exc()

        raise ValueError("Could not load annotations. Check your dataset configuration.")


    def get_transform(self, train):
        """
        Create a composition of image transformations

        Args:
            train (bool): Whether in training or validation mode

        Returns:
            transforms.Compose: Composition of transforms
        """
        transforms = []
        transforms.append(T.ToTensor())

        if train:
            transforms.extend([
                # Color jittering
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),

                # Random horizontal flip
                T.RandomHorizontalFlip(0.5),

                # Random rotation
                T.RandomRotation(10),

                # Random perspective transform
                T.RandomPerspective(distortion_scale=0.2, p=0.5),

                # Normalize
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            # Validation transforms
            transforms.append(
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )

        return T.Compose(transforms)

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        """
        Enhanced get item method with robust image loading and augmentation support
        """
        try:
            img_id = self.image_ids[idx]

            # Possible image file extensions
            possible_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']

            # Flexible filename search patterns
            search_patterns = [
                # Exact match
                img_id,
                f'*{img_id}',

                # Roboflow pattern with hash
                f'*{img_id.replace(".jpg", ".rf.*")}.jpg',
                f'*{img_id.replace(".jpg", "")}.rf.*.jpg',

                # Partial match
                f'*{img_id.split(".")[0]}*'
            ]

            # Debug printing
            print(f"\nSearching for image with ID: {img_id}")
            print(f"Current images directory: {self.images_dir}")
            print("Search Patterns:")
            for pattern in search_patterns:
                print(f"  - {pattern}")

            # Find image path
            img_path = None
            for pattern in search_patterns:

                matching_files = [
                    f for f in self.images_dir.glob(pattern)
                    if f.suffix.lower() in possible_extensions
                       and img_id in f.name  # Ensure the original img_id is in the filename
                ]
                if matching_files:
                    print(f"Matching files found: {matching_files}")
                    img_path = matching_files[0]
                    break

            # Raise error if no image found
            if not img_path or not img_path.exists():
                raise FileNotFoundError(f"No image found for ID: {img_id}")

            # Robust image loading
            image = self._robust_image_load(img_path)

            # Get annotations
            annotation = self.annotations.get(img_id, [])

            # Prepare boxes and labels
            boxes = []
            labels = []

            for ann in annotation:
                # Validate annotation
                if 'bbox' in ann and 'class' in ann:
                    boxes.append(ann['bbox'])
                    labels.append(self.class_to_idx.get(ann['class'], 0))

            # Convert to tensors
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

            # Prepare target dict
            target = {
                'boxes': boxes,
                'labels': labels,
                'image_id': torch.tensor([idx])
            }

            # Apply transforms
            if self.transforms is not None:
                image = self.transforms(image)

            return image, target

        except Exception as e:
            self.logger.error(f"Error processing image at index {idx}: {e}")
            raise

@dataclass
class TrainingConfig:
    # Dataset parameters
    train_data_dir: str
    val_data_dir: str
    image_size: int = 800
    num_workers: int = 4

    # Model parameters
    num_classes: int = 21  # 20 VOC classes + background
    pretrained: bool = True
    backbone: str = 'resnet101'

    # Training parameters
    num_epochs: int = 50
    batch_size: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.0005
    momentum: float = 0.9

    # Learning rate scheduling
    lr_scheduler: str = 'onecycle'  # ['step', 'cosine', 'onecycle']
    lr_steps: List[int] = None
    lr_gamma: float = 0.1
    warmup_epochs: int = 3

    # Loss parameters
    focal_loss_gamma: float = 2.0
    focal_loss_alpha: float = 0.25

    # Augmentation parameters
    aug_scale: tuple = (0.8, 1.2)
    aug_rotate: int = 10

    # Early stopping
    early_stopping_patience: int = 5

    # Training devices
    device: str = 'cpu'

    # Add steps_per_epoch attribute
    steps_per_epoch: int = None

    # Logging and checkpointing
    log_interval: int = 100
    checkpoint_dir: str = './checkpoints'
    experiment_name: str = 'improved_faster_rcnn'

    # Mixed precision training
    use_amp: bool = False

    # Distributed training
    distributed: bool = False
    world_size: int = 1
    dist_url: str = 'env://'

    def __post_init__(self):
        if self.lr_steps is None:
            self.lr_steps = [30, 40]

class ObjectDetectionTrainer:
    def __init__(self, model: nn.Module, config: TrainingConfig):
        self.model = model
        self.config = config
        self.device = torch.device('cpu')

        # Early Stopping setup
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.max_patience = self.config.early_stopping_patience

        # Scaler is not needed for CPU
        self.scaler = None

        self.setup_training()

    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        log_interval = self.config.log_interval

        for batch_idx, (images, targets) in enumerate(tqdm(train_loader)):
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # Regular forward pass
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Regular backward pass
            self.optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            self.optimizer.step()

            if self.config.lr_scheduler == 'onecycle':
                self.scheduler.step()

            total_loss += losses.item()

            if batch_idx % log_interval == 0:
                self.logger.info(
                    f'Train Epoch: {epoch} [{batch_idx}/{len(train_loader)} '
                    f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {losses.item():.6f}'
                )

        return total_loss / len(train_loader)

    @torch.no_grad()
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0

        for images, targets in tqdm(val_loader, desc='Validation'):
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # Regular forward pass
            loss_dict = self.model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            total_loss += losses.item()

        return total_loss / len(val_loader)

    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
        }

        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True, parents=True)

        checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)

        if is_best:
            best_path = checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)

    def setup_training(self):
        """Setup optimizer, scheduler, and other training components"""
        # Setup optimizer with weight decay separation
        parameters = self._get_parameters_with_weight_decay()
        self.optimizer = torch.optim.AdamW(
            parameters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )

        # Setup learning rate scheduler
        self.scheduler = self._setup_scheduler()

        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def train(self, train_loader, val_loader):
        """Full training loop with early stopping"""
        start_time = time.time()

        # Create checkpoint directory if it doesn't exist
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True, parents=True)

        # Configure steps_per_epoch for OneCycleLR if needed
        if self.config.lr_scheduler == 'onecycle':
            self.config.steps_per_epoch = len(train_loader)
            self.scheduler = self._setup_scheduler()  # Recreate scheduler with correct steps

        self.logger.info(f"Starting training for {self.config.num_epochs} epochs...")

        for epoch in range(self.config.num_epochs):
            epoch_start_time = time.time()

            # Training phase
            train_loss = self.train_epoch(train_loader, epoch)

            # Validation phase
            val_loss = self.validate(val_loader)

            # Learning rate scheduling
            if self.config.lr_scheduler in ['step', 'cosine']:
                self.scheduler.step()

            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time

            # Logging
            self.logger.info(
                f'Epoch: {epoch:02d}/{self.config.num_epochs - 1} | '
                f'Train Loss: {train_loss:.6f} | '
                f'Val Loss: {val_loss:.6f} | '
                f'Time: {epoch_time:.2f}s | '
                f'LR: {self.optimizer.param_groups[0]["lr"]:.6f}'
            )

            # Early Stopping Check
            early_stop = self._check_early_stopping(val_loss)
            if early_stop:
                self.logger.info(f"Early stopping triggered at epoch {epoch}")
                break

            # Save checkpoint
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.logger.info(f'New best validation loss: {self.best_val_loss:.6f}')

            self.save_checkpoint(
                epoch=epoch,
                is_best=is_best
            )

        total_time = time.time() - start_time
        self.logger.info(f'Training completed in {total_time / 60:.2f} minutes')
        self.logger.info(f'Best validation loss: {self.best_val_loss:.6f}')

        return self.best_val_loss

    def _check_early_stopping(self, val_loss):
        """
        Check for early stopping condition

        Args:
            val_loss (float): Current validation loss

        Returns:
            bool: Whether to stop training
        """
        if val_loss < self.best_val_loss:
            self.patience_counter = 0
            self.best_val_loss = val_loss
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.max_patience:
                return True
        return False

    def train_all(self, train_loader, val_loader):
        """
        Train all components of the model, including feature extraction
        and object detection heads
        """
        try:
            # Train the model
            best_val_loss = self.train(train_loader, val_loader)
            return best_val_loss

        except Exception as e:
            self.logger.error(f"Training failed: {str(e)}")
            raise

    def _get_parameters_with_weight_decay(self):
        """Separate parameters that should and shouldn't have weight decay"""
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.BatchNorm2d)

        for mn, m in self.model.named_modules():
            for pn, p in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn

                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.model.named_parameters()}

        return [
            {'params': [param_dict[pn] for pn in sorted(list(decay))],
             'weight_decay': self.config.weight_decay},
            {'params': [param_dict[pn] for pn in sorted(list(no_decay))],
             'weight_decay': 0.0}
        ]

    def _setup_scheduler(self):
        """Setup learning rate scheduler based on config"""
        if self.config.lr_scheduler == 'step':
            return torch.optim.lr_scheduler.MultiStepLR(
                self.optimizer,
                milestones=self.config.lr_steps,
                gamma=self.config.lr_gamma
            )
        elif self.config.lr_scheduler == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs
            )
        elif self.config.lr_scheduler == 'onecycle':
            # If steps_per_epoch is not set, use a default or skip OneCycleLR
            if self.config.steps_per_epoch is None:
                self.logger.warning("steps_per_epoch not set. Falling back to default scheduler.")
                return torch.optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=30,
                    gamma=0.1
                )

            return torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                epochs=self.config.num_epochs,
                steps_per_epoch=self.config.steps_per_epoch,
                pct_start=0.3
            )

# Class Imbalance Handling
class ClassBalancer:
    @staticmethod
    def compute_class_weights(ground_truth: Dict) -> Dict[str, float]:
        """
        Compute class weights to address class imbalance

        Args:
            ground_truth (Dict): Ground truth annotations

        Returns:
            Dict[str, float]: Normalized class weights
        """
        # Count total annotations per class
        class_counts = defaultdict(int)
        total_annotations = 0

        for annotations in ground_truth.values():
            for ann in annotations:
                class_counts[ann['class']] += 1
                total_annotations += 1

        # Compute inverse frequency weights
        class_weights = {}
        for cls, count in class_counts.items():
            # Compute weight as: (1 / frequency) * (total classes / total classes + 1)
            class_weights[cls] = (total_annotations / (len(class_counts) * count)) * (
                        len(class_counts) / (len(class_counts) + 1))

        # Normalize weights
        max_weight = max(class_weights.values())
        class_weights = {cls: weight / max_weight for cls, weight in class_weights.items()}

        return class_weights


# Advanced Metrics Computation
class AdvancedMetrics:
    @staticmethod
    def compute_precision_recall_f1(predictions: List[Dict],
                                    ground_truth: List[Dict],
                                    iou_threshold: float = 0.5) -> Dict:
        """
        Compute precision, recall, and F1 score

        Args:
            predictions (List[Dict]): Model predictions
            ground_truth (List[Dict]): Ground truth annotations
            iou_threshold (float): Intersection over Union threshold

        Returns:
            Dict: Precision, Recall, and F1 Score metrics
        """
        # Group predictions and ground truth by class
        pred_by_class = defaultdict(list)
        gt_by_class = defaultdict(list)

        for pred in predictions:
            pred_by_class[pred['class_name']].append(pred)

        for gt in ground_truth:
            gt_by_class[gt['class']].append(gt)

        # Compute metrics per class
        class_metrics = {}
        for cls in set(list(pred_by_class.keys()) + list(gt_by_class.keys())):
            class_preds = pred_by_class.get(cls, [])
            class_gts = gt_by_class.get(cls, [])

            # Sort predictions by confidence
            class_preds.sort(key=lambda x: x['score'], reverse=True)

            tp = fp = fn = 0
            for pred in class_preds:
                # Check if prediction matches any ground truth
                matched = any(
                    calculate_iou(pred['bbox'], gt['bbox']) >= iou_threshold
                    for gt in class_gts
                )

                if matched:
                    tp += 1
                else:
                    fp += 1

            fn = max(0, len(class_gts) - tp)

            # Compute precision, recall, F1
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            class_metrics[cls] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'tp': tp,
                'fp': fp,
                'fn': fn
            }

        return class_metrics

    @staticmethod
    def visualize_class_performance(class_metrics: Dict):
        """
        Visualize class-wise performance metrics

        Args:
            class_metrics (Dict): Performance metrics per class
        """
        # Prepare data for visualization
        classes = list(class_metrics.keys())
        precisions = [metrics['precision'] for metrics in class_metrics.values()]
        recalls = [metrics['recall'] for metrics in class_metrics.values()]
        f1_scores = [metrics['f1_score'] for metrics in class_metrics.values()]

        # Create visualization
        plt.figure(figsize=(12, 6))
        x = np.arange(len(classes))
        width = 0.25

        plt.bar(x - width, precisions, width, label='Precision', color='blue', alpha=0.7)
        plt.bar(x, recalls, width, label='Recall', color='green', alpha=0.7)
        plt.bar(x + width, f1_scores, width, label='F1 Score', color='red', alpha=0.7)

        plt.xlabel('Classes')
        plt.ylabel('Score')
        plt.title('Class-wise Performance Metrics')
        plt.xticks(x, classes, rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig('class_performance_metrics.png')
        plt.close()


# Confidence Calibration
class ConfidenceCalibrator:
    @staticmethod
    def temperature_scaling(logits, temperature=1.0):
        """
        Temperature scaling for confidence calibration

        Args:
            logits (torch.Tensor): Model logits
            temperature (float): Calibration temperature

        Returns:
            torch.Tensor: Calibrated probabilities
        """
        return torch.softmax(logits / temperature, dim=1)

    @staticmethod
    def reliability_diagram(confidences, accuracies, num_bins=10):
        """
        Create reliability diagram (confidence vs accuracy)

        Args:
            confidences (List[float]): Model confidences
            accuracies (List[bool]): Prediction accuracies
            num_bins (int): Number of confidence bins
        """
        plt.figure(figsize=(10, 8))

        # Compute bin edges
        bin_edges = np.linspace(0, 1, num_bins + 1)

        # Compute bin-wise accuracy and confidence
        bin_accuracies = []
        bin_confidences = []

        for i in range(num_bins):
            bin_mask = (confidences >= bin_edges[i]) & (confidences < bin_edges[i + 1])
            if np.sum(bin_mask) > 0:
                bin_accuracies.append(np.mean(accuracies[bin_mask]))
                bin_confidences.append(np.mean(confidences[bin_mask]))
            else:
                bin_accuracies.append(0)
                bin_confidences.append((bin_edges[i] + bin_edges[i + 1]) / 2)

        # Plot reliability diagram
        plt.plot(bin_confidences, bin_accuracies, marker='o')
        plt.plot([0, 1], [0, 1], linestyle='--', color='red')  # Ideal line

        plt.title('Reliability Diagram')
        plt.xlabel('Confidence')
        plt.ylabel('Accuracy')
        plt.savefig('reliability_diagram.png')
        plt.close()


# [Previous classes remain the same: ClassBalancer, AdvancedMetrics, ConfidenceCalibrator]
# Utility function for IoU calculation
def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate Intersection over Union between two boxes"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return intersection / (area1 + area2 - intersection)


class AttentionModule(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 8, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        attention_weights = self.attention(x)
        return x * attention_weights


class ImprovedFasterRCNN(nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super().__init__()

        # Updated model loading method
        if pretrained:
            weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
        else:
            self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)

        # Replace the classifier
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Add attention modules to FPN levels
        self.fpn_attention = nn.ModuleList([
            AttentionModule(256) for _ in range(5)
        ])

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        original_features = self.model.backbone(images)

        # Apply attention to FPN features
        features = {}
        for k, v in original_features.items():
            level = int(k[-1]) - 1
            features[k] = self.fpn_attention[level](v)

        if self.training:
            losses = self.model.rpn(images, features, targets)
            detector_losses = self.model.roi_heads(features, targets)
            losses.update(detector_losses)
            return losses
        else:
            detections = self.model.rpn(images, features)
            detections = self.model.roi_heads(features, detections)
            return detections

class OptimizedObjectDetectionTest:
    def __init__(self,
                 tf_model_dir: str,
                 voc_data_dir: str,
                 num_images: int = 100,
                 iou_threshold: float = 0.5):
        """
        Enhanced Object Detection Test Framework

        Args:
            tf_model_dir (str): TensorFlow model directory
            voc_data_dir (str): VOC dataset directory
            num_images (int): Number of images to test
            iou_threshold (float): IoU threshold for detection
        """
        # Logging setup
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('object_detection_optimization.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Directories and parameters
        self.results_dir = Path("./optimized_results")
        self.results_dir.mkdir(exist_ok=True)

        # Core parameters
        self.iou_threshold = iou_threshold
        self.num_images = num_images

        # Set voc_data_dir explicitly
        self.voc_data_dir = Path(voc_data_dir)

        # Load ground truth and compute class weights
        self.ground_truth = self._load_ground_truth()

        # Add a check to handle empty ground truth
        if not self.ground_truth:
            self.logger.warning("No ground truth found. Using default class weights.")
            # Provide a default class weight if no ground truth
            self.class_weights = {
                'person': 1.0, 'aeroplane': 1.0, 'bicycle': 1.0,
                'bird': 1.0, 'boat': 1.0, 'bottle': 1.0,
                'bus': 1.0, 'car': 1.0, 'cat': 1.0,
                'chair': 1.0, 'cow': 1.0, 'diningtable': 1.0,
                'dog': 1.0, 'horse': 1.0, 'motorbike': 1.0,
                'pottedplant': 1.0, 'sheep': 1.0, 'sofa': 1.0,
                'train': 1.0, 'tvmonitor': 1.0
            }
        else:
            self.class_weights = ClassBalancer.compute_class_weights(self.ground_truth)

        # Log class weights
        self.logger.info("Computed Class Weights:")
        for cls, weight in self.class_weights.items():
            self.logger.info(f"{cls}: {weight:.4f}")

        # Model and data setup
        self.class_names = self._load_class_names()
        self.pytorch_model = self._load_pytorch_model()
        self.tf_model = self._load_tensorflow_model(tf_model_dir)

        # Performance tracking
        self.performance_metrics = {
            'pytorch': defaultdict(list),
            'tensorflow': defaultdict(list)
        }

    def initialize_training(self):
        """Initialize the training pipeline"""
        config = TrainingConfig(
            train_data_dir=str(self.voc_data_dir / 'train'),
            val_data_dir=str(self.voc_data_dir / 'valid'),
            # Temporarily set a default value for steps_per_epoch
            steps_per_epoch=100  # This will be updated in the main training loop
        )

        trainer = ObjectDetectionTrainer(
            model=self.pytorch_model,
            config=config
        )

        return trainer

    def _load_ground_truth(self) -> Dict:
        """
        Load ground truth annotations from JSON file

        Returns:
            Dict: Ground truth annotations
        """
        try:
            # Check multiple possible paths for ground truth file
            possible_paths = [
                self.voc_data_dir / "ground_truth.json",
                self.voc_data_dir / "annotations" / "ground_truth.json",
                self.voc_data_dir / "labels" / "ground_truth.json"
            ]

            ground_truth = {}
            for gt_json_path in possible_paths:
                if gt_json_path.exists():
                    self.logger.info(f"Loading ground truth from: {gt_json_path}")

                    with open(gt_json_path, 'r') as f:
                        ground_truth = json.load(f)

                    break

            if not ground_truth:
                self.logger.error("No ground truth file found in expected locations.")
                return {}

            # Basic validation
            self.logger.info(f"Loaded {len(ground_truth)} ground truth entries")

            # Check first few entries
            sample_keys = list(ground_truth.keys())[:5]
            self.logger.info("Sample ground truth keys:")
            for key in sample_keys:
                self.logger.info(f"  {key}: {len(ground_truth[key])} annotations")

            return ground_truth

        except Exception as e:
            self.logger.error(f"Error loading ground truth: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {}

    def run_comprehensive_tests(self):
        """
        Comprehensive object detection test with advanced optimizations
        """
        self.logger.info("Starting Comprehensive Object Detection Tests")

        # Collect image files
        image_files = self._collect_image_files()

        # Initialize performance tracking
        all_predictions = {
            'pytorch': [],
            'tensorflow': []
        }
        all_ground_truth = []

        # Process images
        for image_file in tqdm(image_files, desc="Processing Images"):
            # Load image and ground truth
            image, gt_boxes = self._prepare_image_and_ground_truth(image_file)

            if image is None or gt_boxes is None:
                continue

            # Run detections
            pytorch_detections = self._run_pytorch_detection(image)
            tf_detections = self._run_tensorflow_detection(image)

            # Store predictions and ground truth
            all_predictions['pytorch'].extend(pytorch_detections)
            all_predictions['tensorflow'].extend(tf_detections)
            all_ground_truth.extend(gt_boxes)

        # Compute advanced metrics
        self._compute_comprehensive_metrics(all_predictions, all_ground_truth)

    def _compute_comprehensive_metrics(self, all_predictions: Dict, all_ground_truth: List):
        """
        Compute comprehensive performance metrics

        Args:
            all_predictions (Dict): Predictions from both models
            all_ground_truth (List): Ground truth annotations
        """
        for model_type in ['pytorch', 'tensorflow']:
            # Compute advanced metrics
            try:
                class_metrics = AdvancedMetrics.compute_precision_recall_f1(
                    all_predictions[model_type],
                    all_ground_truth
                )

                # Visualize class performance
                AdvancedMetrics.visualize_class_performance(class_metrics)

                # Extract confidences and accuracies for reliability diagram
                confidences = [pred['score'] for pred in all_predictions[model_type]]
                accuracies = [
                    any(
                        calculate_iou(pred['bbox'], gt['bbox']) >= self.iou_threshold
                        for gt in all_ground_truth
                    )
                    for pred in all_predictions[model_type]
                ]

                # Create reliability diagram
                ConfidenceCalibrator.reliability_diagram(
                    np.array(confidences),
                    np.array(accuracies)
                )

            except Exception as e:
                self.logger.error(f"Error computing metrics for {model_type} model: {str(e)}")
                import traceback
                self.logger.error(traceback.format_exc())

        # Save detailed analysis
        try:
            self._save_comprehensive_results(all_predictions, all_ground_truth)
        except Exception as e:
            self.logger.error(f"Error saving comprehensive results: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())

    def _save_comprehensive_results(self, all_predictions: Dict, all_ground_truth: List):
        """
        Save comprehensive test results

        Args:
            all_predictions (Dict): Predictions from both models
            all_ground_truth (List): Ground truth annotations
        """
        comprehensive_results = {
            'class_weights': self.class_weights,
            'predictions': {
                model: [
                    {
                        'bbox': pred['bbox'],
                        'class': pred['class_name'],
                        'score': pred['score']
                    } for pred in predictions
                ] for model, predictions in all_predictions.items()
            },
            'ground_truth': all_ground_truth
        }

        # Save to JSON
        with open(self.results_dir / 'comprehensive_results.json', 'w') as f:
            json.dump(comprehensive_results, f, indent=4)

        self.logger.info("Comprehensive results saved successfully.")

    def _collect_image_files(self) -> List[Path]:
        """
        Collect image files for testing

        Returns:
            List[Path]: List of image file paths
        """
        image_files = []
        for split in ['train', 'valid']:
            split_dir = self.voc_data_dir / split
            image_files.extend(list(split_dir.glob('*.jpg')))

        # Limit to specified number of images
        return image_files[:self.num_images]


    def _prepare_image_and_ground_truth(self, image_file: Path) -> Tuple[Optional[np.ndarray], Optional[List[Dict]]]:
        """
        Prepare image and corresponding ground truth

        Args:
            image_file (Path): Path to image file

        Returns:
            Tuple[Optional[np.ndarray], Optional[List[Dict]]]: Image and ground truth
        """
        try:
            # Read image
            image = cv2.imread(str(image_file))
            if image is None:
                self.logger.warning(f"Could not read image: {image_file}")
                return None, None

            # Convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Find corresponding ground truth
            image_filename = image_file.name

            # Try different ways to match ground truth
            gt_boxes = (
                    self.ground_truth.get(image_filename) or
                    self.ground_truth.get(image_filename.replace('.jpg', '')) or
                    self.ground_truth.get(image_filename.split('.')[0])
            )

            if gt_boxes is None:
                self.logger.warning(f"No ground truth found for image: {image_filename}")
                return None, None

            # Convert ground truth to required format if necessary
            processed_gt_boxes = []
            for box in gt_boxes:
                processed_gt_boxes.append({
                    'class': box['class'],
                    'bbox': box['bbox']
                })

            return image_rgb, processed_gt_boxes

        except Exception as e:
            self.logger.error(f"Error preparing image and ground truth: {str(e)}")
            return None, None

    def _load_class_names(self) -> List[str]:
        """
        Load VOC class names

        Returns:
            List[str]: List of class names
        """
        # Standard VOC classes
        voc_classes = [
            'background',  # 0
            'aeroplane',  # 1
            'bicycle',  # 2
            'bird',  # 3
            'boat',  # 4
            'bottle',  # 5
            'bus',  # 6
            'car',  # 7
            'cat',  # 8
            'chair',  # 9
            'cow',  # 10
            'diningtable',  # 11
            'dog',  # 12
            'horse',  # 13
            'motorbike',  # 14
            'person',  # 15
            'pottedplant',  # 16
            'sheep',  # 17
            'sofa',  # 18
            'train',  # 19
            'tvmonitor'  # 20
        ]

        self.logger.info(f"Loaded {len(voc_classes)} VOC classes")
        return voc_classes

    def _load_pytorch_model(self) -> torch.nn.Module:
        """
        Load improved PyTorch object detection model configured for PASCAL VOC

        Returns:
            torch.nn.Module: Loaded PyTorch model
        """
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.logger.info(f"Using {device} for PyTorch model")

            # Create improved model
            num_classes = 21  # 20 VOC classes + background
            model = ImprovedFasterRCNN(num_classes=num_classes, pretrained=True)

            # Move to device and set to eval mode
            model.to(device)
            model.eval()

            self.logger.info("Improved PyTorch model loaded and configured for VOC dataset")
            return model

        except Exception as e:
            self.logger.error(f"Error loading PyTorch model: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise

    def _load_tensorflow_model(self, model_dir: str):
        """
        Load TensorFlow object detection model

        Args:
            model_dir (str): Directory containing the saved TensorFlow model

        Returns:
            Loaded TensorFlow model
        """
        try:
            self.logger.info(f"Loading TensorFlow model from {model_dir}")

            # Suppress TensorFlow logging
            tf.get_logger().setLevel('ERROR')

            # Load the model using tf.saved_model.load()
            model = tf.saved_model.load(model_dir)

            self.logger.info("TensorFlow model loaded successfully")
            return model

        except Exception as e:
            self.logger.error(f"Error loading TensorFlow model: {str(e)}")
            raise

    def _run_pytorch_detection(self, image: np.ndarray) -> List[Dict]:
        """
        Run object detection using PyTorch model

        Args:
            image (np.ndarray): Input image

        Returns:
            List[Dict]: Detected objects with their properties
        """
        try:
            # VOC class names (matching model's class indices)
            voc_classes = [
                'background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
                'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
                'train', 'tvmonitor'
            ]

            # Prepare image for PyTorch model
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

            # Convert image to tensor and add batch dimension
            input_tensor = transform(image).unsqueeze(0)

            # Run inference
            with torch.no_grad():
                predictions = self.pytorch_model(input_tensor)

            # Process detections
            processed_detections = []

            # Ensure predictions have expected structure
            if not predictions or len(predictions[0]['boxes']) == 0:
                self.logger.warning("No boxes detected by PyTorch model")
                return []

            # Lower confidence threshold to detect more objects
            confidence_threshold = 0.3

            for i in range(len(predictions[0]['boxes'])):
                try:
                    score = predictions[0]['scores'][i].item()

                    # Filter detections based on confidence threshold
                    if score > confidence_threshold:
                        bbox = predictions[0]['boxes'][i].tolist()
                        class_id = predictions[0]['labels'][i].item()

                        # Ensure class_id is within valid range
                        if 0 <= class_id < len(voc_classes):
                            processed_detections.append({
                                'bbox': bbox,
                                'score': score,
                                'class_name': voc_classes[class_id],
                                'class_id': class_id
                            })

                except (IndexError, TypeError) as e:
                    self.logger.warning(f"Error processing detection at index {i}: {str(e)}")
                    continue

            self.logger.info(f"Processed {len(processed_detections)} PyTorch detections")
            return processed_detections

        except Exception as e:
            self.logger.error(f"PyTorch detection error: {str(e)}")
            import traceback
            self.logger.error(traceback.format_exc())
            return []

    def _run_tensorflow_detection(self, image: np.ndarray) -> List[Dict]:
        """
        Run object detection using TensorFlow model

        Args:
            image (np.ndarray): Input image

        Returns:
            List[Dict]: Detected objects with their properties
        """
        try:
            # Prepare input tensor
            input_tensor = tf.convert_to_tensor(image)
            input_tensor = input_tensor[tf.newaxis, ...]

            # Run inference
            detections = self.tf_model(input_tensor)

            # Process detections
            processed_detections = []

            # Safely extract number of detections
            num_detections = int(detections.get('num_detections', [0])[0])

            # Ensure all required keys exist
            if not all(key in detections for key in ['detection_scores', 'detection_boxes', 'detection_classes']):
                self.logger.warning("Missing detection keys in TensorFlow model output")
                return []

            for i in range(num_detections):
                # Safely extract scores, adding checks to prevent index errors
                try:
                    score = float(detections['detection_scores'][0][i])

                    # Filter low confidence detections
                    if score > 0.5:
                        # Get bounding box
                        bbox = detections['detection_boxes'][0][i].numpy()
                        height, width = image.shape[:2]

                        # Convert normalized coordinates to pixel coordinates
                        bbox_pixels = [
                            bbox[1] * width,  # xmin
                            bbox[0] * height,  # ymin
                            bbox[3] * width,  # xmax
                            bbox[2] * height  # ymax
                        ]

                        # Get class ID
                        class_id = int(detections['detection_classes'][0][i])

                        processed_detections.append({
                            'bbox': bbox_pixels,
                            'score': score,
                            'class_name': self.class_names[class_id],
                            'class_id': class_id
                        })

                except (IndexError, TypeError) as e:
                    self.logger.warning(f"Error processing detection at index {i}: {str(e)}")
                    continue

            self.logger.info(f"Processed {len(processed_detections)} TensorFlow detections")
            return processed_detections

        except Exception as e:
            self.logger.error(f"TensorFlow detection error: {str(e)}")
            # Log more detailed error information
            import traceback
            self.logger.error(traceback.format_exc())
            return []

    def generate_accuracy_evaluator_metrics(self) -> Dict:
        """
        Generate metrics in the format required by accuracy_evaluator.py

        Returns:
            Dict: Metrics in the required format for accuracy evaluation
        """
        self.logger.info("Generating metrics for accuracy evaluator")

        # Initialize metrics structure
        frame_metrics = {}

        # Collect image files
        image_files = self._collect_image_files()

        # Process each image
        for image_file in tqdm(image_files, desc="Generating metrics"):
            try:
                # Get frame_id from image filename
                frame_id = image_file.stem  # removes path and extension

                # Load and prepare image
                image, gt_boxes = self._prepare_image_and_ground_truth(image_file)

                if image is None:
                    continue

                # Initialize frame metrics
                frame_metrics[frame_id] = {}

                # PyTorch detections
                start_time = time.time()
                pytorch_detections = self._run_pytorch_detection(image)
                pytorch_inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds

                # Convert PyTorch boxes to [x, y, width, height] format
                pytorch_boxes = []
                for det in pytorch_detections:
                    box = det['bbox']
                    # Convert [x1, y1, x2, y2] to [x, y, width, height]
                    width = box[2] - box[0]
                    height = box[3] - box[1]
                    pytorch_boxes.append([box[0], box[1], width, height])

                # Store PyTorch metrics
                frame_metrics[frame_id]['pytorch'] = {
                    'inference_time': pytorch_inference_time,
                    'num_detections': len(pytorch_detections),
                    'boxes': pytorch_boxes
                }

                # TensorFlow detections
                start_time = time.time()
                tf_detections = self._run_tensorflow_detection(image)
                tf_inference_time = (time.time() - start_time) * 1000  # Convert to milliseconds

                # Convert TensorFlow boxes to [x, y, width, height] format
                tf_boxes = []
                for det in tf_detections:
                    box = det['bbox']
                    # Convert [x1, y1, x2, y2] to [x, y, width, height]
                    width = box[2] - box[0]
                    height = box[3] - box[1]
                    tf_boxes.append([box[0], box[1], width, height])

                # Store TensorFlow metrics
                frame_metrics[frame_id]['tensorflow'] = {
                    'inference_time': tf_inference_time,
                    'num_detections': len(tf_detections),
                    'boxes': tf_boxes
                }

            except Exception as e:
                self.logger.error(f"Error processing image {image_file}: {str(e)}")
                continue

        # Save metrics to file
        output_metrics = {'frame_metrics': frame_metrics}

        metrics_file = self.results_dir / 'final_metrics_with_frames.json'
        try:
            with open(metrics_file, 'w') as f:
                json.dump(output_metrics, f, indent=4)
            self.logger.info(f"Metrics saved to: {metrics_file}")
        except Exception as e:
            self.logger.error(f"Error saving metrics: {str(e)}")

        return output_metrics

    # [Previous methods from the previous implementation remain the same]

    def run_comprehensive_tests(self):
        """
        Comprehensive object detection test with advanced optimizations
        """
        self.logger.info("Starting Comprehensive Object Detection Tests")

        # Collect image files
        image_files = self._collect_image_files()

        # Initialize performance tracking
        all_predictions = {
            'pytorch': [],
            'tensorflow': []
        }
        all_ground_truth = []

        # Process images
        for image_file in tqdm(image_files, desc="Processing Images"):
            # Load image and ground truth
            image, gt_boxes = self._prepare_image_and_ground_truth(image_file)

            if image is None or gt_boxes is None:
                continue

            # Run detections
            pytorch_detections = self._run_pytorch_detection(image)
            tf_detections = self._run_tensorflow_detection(image)

            # Store predictions and ground truth
            all_predictions['pytorch'].extend(pytorch_detections)
            all_predictions['tensorflow'].extend(tf_detections)
            all_ground_truth.extend(gt_boxes)

        # Compute advanced metrics
        self._compute_comprehensive_metrics(all_predictions, all_ground_truth)


def load_config(config_path='config.yaml'):
    """
    Load configuration from a YAML file

    Args:
        config_path (str): Path to the configuration file

    Returns:
        dict: Loaded configuration
    """
    try:
        # Check if config file exists
        if not os.path.exists(config_path):
            logging.warning(f"Config file not found at {config_path}. Using default settings.")
            return {}

        # Load configuration
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        return config

    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        return {}


def validate_paths(config):
    """
    Validate paths in the configuration

    Args:
        config (dict): Configuration dictionary

    Raises:
        ValueError: If directories do not exist
    """
    # Paths to validate
    paths_to_check = [
        config.get('tf_model_dir'),
        config.get('voc_data_dir')
    ]

    # Check paths
    for path in paths_to_check:
        if path and not os.path.exists(path):
            raise ValueError(f"Directory not found: {path}")


def main():
    """
    Main function to run the optimized object detection test and training
    """
    # Enhanced logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('object_detection_training.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Add argument parser for additional flexibility
    parser = argparse.ArgumentParser(description='Object Detection Training')
    parser.add_argument('--config',
                        type=str,
                        default='config.yaml',
                        help='Path to configuration file')
    parser.add_argument('--tf_model_dir',
                        help='Override TensorFlow model directory')
    parser.add_argument('--voc_data_dir',
                        help='Override VOC dataset directory')

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override config with command-line arguments if provided
    if args.tf_model_dir:
        config['tf_model_dir'] = args.tf_model_dir
    if args.voc_data_dir:
        config['voc_data_dir'] = args.voc_data_dir

    # Validate paths
    validate_paths(config)

    try:
        # Check CUDA availability
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")

        # Define paths from config
        tf_model_dir = config.get('tf_model_dir')
        voc_data_dir = config.get('voc_data_dir')

        # Validate data directories
        train_subdir = config.get('train_subdir', 'train')
        valid_subdir = config.get('valid_subdir', 'valid')
        train_data_dir = Path(voc_data_dir) / train_subdir
        val_data_dir = Path(voc_data_dir) / valid_subdir

        if not train_data_dir.exists() or not val_data_dir.exists():
            raise ValueError(f"Training or validation directory not found. Check path: {voc_data_dir}")

        # Create test runner
        test_runner = OptimizedObjectDetectionTest(
            tf_model_dir=tf_model_dir,
            voc_data_dir=voc_data_dir,
            num_images=config.get('num_images', 100),
            iou_threshold=config.get('iou_threshold', 0.5)
        )

        # Initialize training pipeline
        trainer = test_runner.initialize_training()
        logger.info("Training pipeline initialized")

        # Create data loaders with train/val distinction
        '''train_dataset = VOCDetectionDataset(
            data_dir=str(train_data_dir),
            train=True
        )
        val_dataset = VOCDetectionDataset(
            data_dir=str(val_data_dir),
            train=False
        )'''

        # Before creating this block of code:
        train_dataset = VOCDetectionDataset(
            data_dir="/mnt/c/Users/ashut/MachineLearning/SelfAdaptiveObjectDetection/Pascal/VOC/train",
            train=True
        )

        # Add the new code block here:
        print("\n--- Images in Train Directory ---")
        train_dir = Path("/mnt/c/Users/ashut/MachineLearning/SelfAdaptiveObjectDetection/Pascal/VOC/train")
        image_extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        for img in train_dir.glob('*'):
            if img.suffix.lower() in image_extensions:
                print(img.name)
        print(f"\nTotal image files: {len(list(train_dir.glob('*' + ''.join(image_extensions))))}")

        ''' val_dataset = VOCDetectionDataset(
            data_dir="/mnt/c/Users/ashut/MachineLearning/SelfAdaptiveObjectDetection/Pascal/VOC/valid",
            train=False
        '''

        try:
            val_dataset = VOCDetectionDataset(
                data_dir="/mnt/c/Users/ashut/MachineLearning/SelfAdaptiveObjectDetection/Pascal/VOC/valid",
                train=True
            )
        except FileNotFoundError as e:
            print(f"Warning: Validation dataset not found. Skipping validation. Error: {e}")
            val_dataset = None

        # Log dataset sizes
        logger.info(f"Training dataset size: {len(train_dataset)}")
        if val_dataset is not None:
            logger.info(f"Validation dataset size: {len(val_dataset)}")
        else:
            logger.warning("Validation dataset is None. Skipping validation steps.")
        #logger.info(f"Validation dataset size: {len(val_dataset)}")

        # Create data loaders with config parameters
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.get('batch_size', trainer.config.batch_size),
            shuffle=True,
            num_workers=config.get('num_workers', trainer.config.num_workers),
            collate_fn=collate_fn,
            pin_memory=True  # Improve data transfer to GPU
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.get('batch_size', trainer.config.batch_size),
            shuffle=False,
            num_workers=config.get('num_workers', trainer.config.num_workers),
            collate_fn=collate_fn,
            pin_memory=True
        )

        # Update steps_per_epoch in the trainer's config
        trainer.config.steps_per_epoch = len(train_loader)

        # Additional pre-training validation
        logger.info("Validating data loaders...")
        _validate_data_loaders(train_loader, val_loader, logger)

        # Run training
        logger.info("Starting model training...")
        best_val_loss = trainer.train(train_loader, val_loader)
        logger.info(f"Training completed with best validation loss: {best_val_loss:.4f}")

        # Generate metrics for accuracy evaluator
        metrics = test_runner.generate_accuracy_evaluator_metrics()
        logger.info("Generated metrics for accuracy evaluator")

        # Run comprehensive tests
        test_runner.run_comprehensive_tests()

        logger.info("Object Detection Optimization Complete!")

    except Exception as e:
        logger.error(f"Test execution failed: {str(e)}")
        logger.error(traceback.format_exc())
        # Optionally send an alert or log to a monitoring system


def _validate_data_loaders(train_loader, val_loader, logger):
    """
    Validate data loaders before training
    """
    try:
        # Check if we can iterate through both loaders
        logger.info("Checking training data loader...")
        for batch_idx, (images, targets) in enumerate(train_loader):
            if batch_idx >= 5:  # Check first 5 batches
                break

            # Basic validation
            assert len(images) > 0, "No images in training batch"
            assert len(targets) > 0, "No targets in training batch"

            # Optional: Log batch details
            logger.info(f"Train Batch {batch_idx}: {len(images)} images")

        logger.info("Checking validation data loader...")
        for batch_idx, (images, targets) in enumerate(val_loader):
            if batch_idx >= 5:  # Check first 5 batches
                break

            # Basic validation
            assert len(images) > 0, "No images in validation batch"
            assert len(targets) > 0, "No targets in validation batch"

            # Optional: Log batch details
            logger.info(f"Val Batch {batch_idx}: {len(images)} images")

        logger.info("Data loaders validated successfully")

    except Exception as e:
        logger.error(f"Data loader validation failed: {str(e)}")
        raise


def collate_fn(batch):
    """
    Custom collate function for the data loader
    Handles variable-sized inputs in object detection
    """
    images = []
    targets = []
    for image, target in batch:
        images.append(image)
        targets.append(target)
    return images, targets


if __name__ == "__main__":
    main()