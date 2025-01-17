import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, random_split
from torchvision.models.detection import fasterrcnn_resnet50_fpn, retinanet_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.retinanet import RetinaNetHead
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import logging
import json
import time
from datetime import datetime
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score
import torch.multiprocessing as mp
from test_dataset_loading import DatasetTester
from nhollistic_test import VOCDetectionDataset
from collections import defaultdict
import cv2
from tqdm import tqdm



class EnhancedParallelTrainer:
    def __init__(self, data_dir, model_configs):
        self.data_dir = data_dir
        self.model_configs = model_configs

        # Enforce GPU usage
        if not torch.cuda.is_available():
            raise RuntimeError("This program requires a GPU, but none is available.")

        self.device = torch.device('cuda')

        # Setup logging
        self.setup_logging()

        # Log GPU information
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        self.logger.info(f"Using GPU: {gpu_name} with {gpu_memory:.2f} GB memory")
        self.logger.info(f"Number of GPUs available: {torch.cuda.device_count()}")

        # Initialize metrics storage
        self.metrics = {
            model_type: {
                'train_loss': [],
                'val_map': [],
                'val_f1': [],
                'precision': [],
                'recall': [],
                'inference_times': []
            } for model_type in model_configs.keys()
        }

    def setup_logging(self):
        """Configure logging."""
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"parallel_training_{timestamp}.log")

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def prepare_datasets(self):
        """Prepare and validate datasets using DatasetTester."""
        self.logger.info("Preparing datasets...")

        # Initialize DatasetTester
        tester = DatasetTester(self.data_dir)

        # Load and filter dataset
        dataset = VOCDetectionDataset(data_dir=self.data_dir, train=True)
        filtered_dataset = tester.filter_missing_files(dataset)

        # Split into train/val sets
        train_size = int(0.8 * len(filtered_dataset))
        val_size = len(filtered_dataset) - train_size
        train_dataset, val_dataset = random_split(filtered_dataset, [train_size, val_size])

        # Use batch size from first model config (they're the same in this case)
        batch_size = next(iter(self.model_configs.values()))['batch_size']

        # Create dataloaders
        self.train_loader = tester.prepare_dataloader(
            train_dataset,
            batch_size=batch_size,  # Fixed: Using batch_size from config
            num_workers=4
            #pin_memory=True
        )

        self.val_loader = tester.prepare_dataloader(
            val_dataset,
            batch_size=batch_size,  # Fixed: Using batch_size from config
            num_workers=4
            #pin_memory=True
        )

        self.logger.info(f"Dataset prepared - Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    def _create_model(self, model_type, config):
        """Create model based on type (frcnn or retinanet)."""
        num_classes = config['num_classes']

        if model_type == 'frcnn':
            # Create FRCNN model
            model = fasterrcnn_resnet50_fpn(
                weights=None,  # No COCO weights
                weights_backbone="IMAGENET1K_V1"  # Use ImageNet pretrained backbone
            )
            # Modify classifier head
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        elif model_type == 'retinanet':
            # Create RetinaNet model
            model = retinanet_resnet50_fpn(
                weights=None,  # No COCO weights
                weights_backbone="IMAGENET1K_V1"  # Use ImageNet pretrained backbone
            )
            # Modify head for num_classes
            num_anchors = model.head.classification_head.num_anchors
            in_channels = model.backbone.out_channels
            model.head = RetinaNetHead(
                in_channels=in_channels,
                num_anchors=num_anchors,
                num_classes=num_classes
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Move model to device
        model = model.to(self.device)

        # Enable cudnn benchmarking for better performance
        torch.backends.cudnn.benchmark = True

        self.logger.info(f"Created {model_type} model with {num_classes} classes")
        return model

    def train_models_parallel(self):
        """Train multiple models in parallel."""
        processes = []
        num_gpus = torch.cuda.device_count()
        self.logger.info(f"Starting parallel training with {num_gpus} GPUs")

        # Create a manager for shared memory
        manager = mp.Manager()
        shared_metrics = manager.dict()

        # Initialize shared metrics for each model
        for model_type in self.model_configs.keys():
            shared_metrics[model_type] = manager.dict({
                'train_loss': manager.list(),
                'val_map': manager.list(),
                'val_f1': manager.list(),
                'precision': manager.list(),
                'recall': manager.list(),
                'inference_times': manager.list()
            })

        # Start parallel processes
        for idx, (model_type, config) in enumerate(self.model_configs.items()):
            gpu_id = idx % num_gpus
            self.logger.info(f"Assigning {model_type} to GPU {gpu_id}")

            p = mp.Process(
                target=self._train_single_model,
                args=(model_type, config, gpu_id, shared_metrics[model_type])
            )
            p.start()
            processes.append(p)
            self.logger.info(f"Started training process for {model_type} on GPU {gpu_id}")

        # Wait for all processes to complete
        for p in processes:
            p.join()

        # Copy shared metrics back to instance metrics
        for model_type in self.model_configs.keys():
            for k, v in shared_metrics[model_type].items():
                self.metrics[model_type][k] = list(v)

    def _train_single_model(self, model_type, config, gpu_id, shared_metrics):
        """Train a single model with validation."""
        try:
            # Set GPU device for this process
            torch.cuda.set_device(gpu_id)
            self.logger.info(f"Training {model_type} on GPU {gpu_id}")

            # Initialize model, optimizer, etc.
            model = self._create_model(model_type, config)
            optimizer = SGD(
                model.parameters(),
                lr=config['learning_rate'],
                momentum=0.9
            )
            scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
            scaler = torch.cuda.amp.GradScaler()

            total_epochs = config['num_epochs']
            self.logger.info(f"Starting training for {model_type} - Total epochs: {total_epochs}")

            best_map = 0
            patience = 5
            patience_counter = 0

            for epoch in range(total_epochs):
                self.logger.info(f"\n{'=' * 50}")
                self.logger.info(f"Starting Epoch {epoch + 1}/{total_epochs} for {model_type}")
                self.logger.info(f"{'=' * 50}")

                # Training phase
                train_loss = self._train_epoch(model, optimizer, scaler, model_type, epoch, total_epochs)

                # Update shared metrics
                shared_metrics['train_loss'].append(train_loss)

                # Validation phase
                self.logger.info(f"Starting validation for epoch {epoch + 1}/{total_epochs}")
                val_metrics = self._validate_model(model, model_type)

                # Update shared metrics with validation results
                shared_metrics['val_map'].append(val_metrics['mAP'])
                shared_metrics['inference_times'].append(val_metrics['inference_times'])
                if 'f1' in val_metrics:
                    shared_metrics['val_f1'].append(val_metrics['f1'])
                if 'precision' in val_metrics:
                    shared_metrics['precision'].append(val_metrics['precision'])
                if 'recall' in val_metrics:
                    shared_metrics['recall'].append(val_metrics['recall'])

                # Logging
                self.logger.info(
                    f"\nEpoch {epoch + 1}/{total_epochs} Summary for {model_type}:\n"
                    f"Train Loss: {train_loss:.4f}\n"
                    f"Validation mAP: {val_metrics['mAP']:.4f}\n"
                    f"Best mAP so far: {best_map:.4f}"
                )

                # Save checkpoint and check early stopping
                if val_metrics['mAP'] > best_map:
                    best_map = val_metrics['mAP']
                    self._save_checkpoint(model, optimizer, epoch, model_type, val_metrics)
                    self.logger.info(f"New best mAP achieved! Saving checkpoint")
                    patience_counter = 0
                else:
                    patience_counter += 1
                    self.logger.info(f"No improvement for {patience_counter} epochs")
                    if patience_counter >= patience:
                        self.logger.info(f"Early stopping triggered for {model_type} after {epoch + 1} epochs")
                        break

                scheduler.step()

        except Exception as e:
            self.logger.error(f"Error training {model_type} on GPU {gpu_id}: {str(e)}")
            raise

    def _train_epoch(self, model, optimizer, scaler, model_type, epoch, total_epochs):
        """Train for one epoch with mixed precision."""
        model.train()
        total_loss = 0
        num_batches = len(self.train_loader)

        self.logger.info(f"Training epoch {epoch + 1}/{total_epochs} - Total batches: {num_batches}")

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = [img.to(self.device) for img in images]
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # Mixed precision training
            with torch.cuda.amp.autocast():
                loss_dict = model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += losses.item()

            if batch_idx % 10 == 0:
                progress = (batch_idx + 1) / num_batches * 100
                self.logger.info(
                    f"{model_type} Epoch {epoch + 1}/{total_epochs} "
                    f"[Batch {batch_idx + 1}/{num_batches} ({progress:.1f}%)] "
                    f"Loss: {losses.item():.4f}"
                )

        avg_loss = total_loss / num_batches
        self.logger.info(f"Epoch {epoch + 1}/{total_epochs} completed - Average loss: {avg_loss:.4f}")
        return avg_loss

    def _validate_model(self, model, model_type):
        """Validate model and compute metrics."""
        model.eval()
        metrics = defaultdict(list)
        num_batches = len(self.val_loader)

        self.logger.info(f"Starting validation - Total batches: {num_batches}")

        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(self.val_loader):
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                # Time inference
                start_time = time.time()
                predictions = model(images)
                inference_time = time.time() - start_time

                # Compute metrics
                self._compute_batch_metrics(predictions, targets, metrics)
                metrics['inference_times'].append(inference_time)

                if batch_idx % 10 == 0:
                    progress = (batch_idx + 1) / num_batches * 100
                    self.logger.info(
                        f"Validation progress: {progress:.1f}% "
                        f"[{batch_idx + 1}/{num_batches}]"
                    )

        # Average metrics
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
        self.logger.info(f"Validation completed - Average inference time: {avg_metrics['inference_times']:.4f}s")
        return avg_metrics

    def _compute_batch_metrics(self, predictions, targets, metrics):
        """Compute detection metrics for a batch."""
        for pred, target in zip(predictions, targets):
            # Convert predictions to the format needed for metrics
            pred_boxes = pred['boxes'].cpu().numpy()
            pred_scores = pred['scores'].cpu().numpy()
            pred_labels = pred['labels'].cpu().numpy()

            # Convert target to numpy
            true_boxes = target['boxes'].cpu().numpy()
            true_labels = target['labels'].cpu().numpy()

            # Compute metrics for each class
            for class_id in range(1, self.model_configs[next(iter(self.model_configs))]['num_classes']):
                class_pred_indices = pred_labels == class_id
                class_true_indices = true_labels == class_id

                if not any(class_true_indices):
                    continue

                class_pred_boxes = pred_boxes[class_pred_indices]
                class_pred_scores = pred_scores[class_pred_indices]
                class_true_boxes = true_boxes[class_true_indices]

                # Compute IoU matrix
                if len(class_pred_boxes) > 0 and len(class_true_boxes) > 0:
                    iou_matrix = self._compute_iou_matrix(class_pred_boxes, class_true_boxes)

                    # Compute precision and recall
                    for threshold in [0.5, 0.75, 0.9]:  # Different IoU thresholds
                        tp = np.sum(np.max(iou_matrix, axis=1) >= threshold)
                        fp = len(class_pred_boxes) - tp
                        fn = len(class_true_boxes) - tp

                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0

                        metrics[f'precision_{threshold}'].append(precision)
                        metrics[f'recall_{threshold}'].append(recall)

                    # Compute mAP
                    metrics['mAP'].append(self._compute_ap(iou_matrix, class_pred_scores))

        return metrics

    def _compute_iou_matrix(self, boxes1, boxes2):
        """Compute IoU matrix between two sets of boxes."""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

        lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
        rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])

        wh = np.clip(rb - lt, 0, None)
        inter = wh[:, :, 0] * wh[:, :, 1]

        union = area1[:, None] + area2 - inter

        return inter / union

    def _compute_ap(self, iou_matrix, scores, iou_threshold=0.5):
        """Compute Average Precision given IoU matrix and prediction scores."""
        gt_matches = np.zeros(iou_matrix.shape[1])
        sorted_indices = np.argsort(-scores)
        matched = np.zeros(iou_matrix.shape[1])

        tp = np.zeros(len(sorted_indices))
        fp = np.zeros(len(sorted_indices))

        for idx, pred_idx in enumerate(sorted_indices):
            if np.sum(iou_matrix[pred_idx] >= iou_threshold) == 0:
                fp[idx] = 1
                continue

            gt_idx = np.argmax(iou_matrix[pred_idx])
            if matched[gt_idx]:
                fp[idx] = 1
            else:
                tp[idx] = 1
                matched[gt_idx] = 1

        fp = np.cumsum(fp)
        tp = np.cumsum(tp)

        recalls = tp / float(iou_matrix.shape[1])
        precisions = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

        return np.trapz(precisions, recalls)

    def _save_checkpoint(self, model, optimizer, epoch, model_type, metrics):
        """Save model checkpoint."""
        checkpoint_dir = os.path.join("checkpoints", model_type)
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }

        path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint for {model_type} epoch {epoch}")

    def generate_plots(self):
        """Generate training and evaluation plots."""
        for model_type in self.model_configs.keys():
            self._plot_model_metrics(model_type)

    def _plot_model_metrics(self, model_type):
        """Plot metrics for a single model."""
        plt.figure(figsize=(15, 10))

        # Plot training loss
        plt.subplot(2, 2, 1)
        plt.plot(self.metrics[model_type]['train_loss'])
        plt.title(f'{model_type} Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

        # Plot validation mAP
        plt.subplot(2, 2, 2)
        plt.plot(self.metrics[model_type]['val_map'])
        plt.title(f'{model_type} Validation mAP')
        plt.xlabel('Epoch')
        plt.ylabel('mAP')

        plt.tight_layout()
        plt.savefig(f'{model_type}_metrics.png')
        plt.close()


def main():
    # Configuration
    DATA_DIR = "/mnt/c/Users/ashut/MachineLearning/SelfAdaptiveObjectDetection/Pascal/VOC/train"
    model_configs = {
        'frcnn': {
            'num_classes': 21,
            'learning_rate': 0.005,
            'batch_size': 4,
            'num_epochs': 50
        },
        'retinanet': {
            'num_classes': 21,
            'learning_rate': 0.001,
            'batch_size': 4,
            'num_epochs': 50
        }
    }

    # Initialize trainer
    trainer = EnhancedParallelTrainer(DATA_DIR, model_configs)

    # Prepare datasets
    trainer.prepare_datasets()

    # Train models in parallel
    trainer.train_models_parallel()

    # Generate plots
    trainer.generate_plots()


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()