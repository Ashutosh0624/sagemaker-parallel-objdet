import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
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
from metrics_analyzer import PublicationMetricsAnalyzer
from collections import defaultdict
import numpy as np

class EnhancedParallelTrainer:
    def __init__(self, data_dir, model_configs, rank, world_size, output_dir):
        """
        Initialize the enhanced parallel trainer.
        
        Args:
            data_dir (str): Directory containing training data
            model_configs (dict): Configuration for each model type
            rank (int): Current process rank
            world_size (int): Total number of processes
            output_dir (str): Directory for saving outputs
        """
        self.data_dir = data_dir
        self.model_configs = model_configs
        self.rank = rank
        self.world_size = world_size
        self.output_dir = output_dir
        
        # Set device
        self.device = torch.device(f'cuda:{rank}')
        
        # Initialize metrics analyzer on rank 0
        if self.rank == 0:
            self.metrics_analyzer = PublicationMetricsAnalyzer(output_dir)
            self.setup_logging()
        
        # Initialize training history and metrics storage
        self.initialize_metrics_storage()
        
        # Enable cuDNN benchmarking for better performance
        torch.backends.cudnn.benchmark = True

    def setup_logging(self):
        """Setup logging configuration."""
        log_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f"training_{timestamp}.log")
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def initialize_metrics_storage(self):
        """Initialize storage for training metrics and history."""
        self.training_history = {
            model_type: {
                'epoch': [],
                'train_loss': [],
                'train_mAP': [],
                'train_f1': [],
                'val_loss': [],
                'val_mAP': [],
                'val_f1': [],
                'learning_rate': [],
                'inference_times': []
            } for model_type in self.model_configs.keys()
        }
        
        self.evaluation_metrics = {
            model_type: {
                'final_metrics': None,
                'per_class_metrics': {},
                'inference_stats': {},
                'model_stats': {}
            } for model_type in self.model_configs.keys()
        }

    def prepare_datasets(self):
        """Prepare datasets with distributed sampling."""
        from voc_dataset import VOCDetectionDataset
        
        # Load dataset
        full_dataset = VOCDetectionDataset(data_dir=self.data_dir, train=True)
        
        # Split into train/val sets
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
        
        # Create distributed samplers
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=self.world_size,
            rank=self.rank
        )
        
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=self.world_size,
            rank=self.rank
        )
        
        # Create dataloaders
        batch_size = next(iter(self.model_configs.values()))['batch_size']
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=4,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            num_workers=4,
            pin_memory=True,
            collate_fn=self._collate_fn
        )
        
        if self.rank == 0:
            self.logger.info(
                f"Datasets prepared - Train: {len(train_dataset)}, "
                f"Val: {len(val_dataset)}, "
                f"Batch size per GPU: {batch_size}"
            )

    @staticmethod
    def _collate_fn(batch):
        """Custom collate function for object detection."""
        return tuple(zip(*batch))

    def _create_model(self, model_type, config):
        """Create and wrap model in DistributedDataParallel."""
        num_classes = config['num_classes']
        
        if model_type == 'frcnn':
            model = fasterrcnn_resnet50_fpn(
                weights=None,
                weights_backbone="IMAGENET1K_V1"
            )
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            
        elif model_type == 'retinanet':
            model = retinanet_resnet50_fpn(
                weights=None,
                weights_backbone="IMAGENET1K_V1"
            )
            num_anchors = model.head.classification_head.num_anchors
            in_channels = model.backbone.out_channels
            model.head = RetinaNetHead(
                in_channels=in_channels,
                num_anchors=num_anchors,
                num_classes=num_classes
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Move model to device and wrap with DDP
        model = model.to(self.device)
        model = DistributedDataParallel(
            model,
            device_ids=[self.rank],
            output_device=self.rank
        )
        
        if self.rank == 0:
            self.logger.info(f"Created {model_type} model with {num_classes} classes")
        
        return model

    def train_models_parallel(self):
        """Train multiple models in sequence with distributed training."""
        for model_type, config in self.model_configs.items():
            if self.rank == 0:
                self.logger.info(f"\nStarting training for {model_type}")
            
            # Train model
            model = self._create_model(model_type, config)
            self._train_single_model(model_type, config, model)
            
            # Save results on rank 0
            if self.rank == 0:
                self.save_training_results(model_type)
            
            # Synchronize processes
            dist.barrier()

    def _train_single_model(self, model_type, config, model):
        """Train a single model with validation."""
        optimizer = SGD(
            model.parameters(),
            lr=config['learning_rate'],
            momentum=0.9
        )
        scheduler = StepLR(optimizer, step_size=3, gamma=0.1)
        scaler = torch.cuda.amp.GradScaler()
        
        best_map = 0
        patience = 5
        patience_counter = 0
        
        for epoch in range(config['num_epochs']):
            if self.rank == 0:
                self.logger.info(f"Starting epoch {epoch + 1}/{config['num_epochs']}")
            
            # Training phase
            train_loss = self._train_epoch(model, optimizer, scaler, model_type, epoch)
            
            # Validation phase
            val_metrics = self._validate_model(model, model_type)
            
            # Update metrics and check early stopping on rank 0
            if self.rank == 0:
                self._update_metrics(model_type, epoch, train_loss, val_metrics)
                
                if val_metrics['mean_metrics']['mAP'] > best_map:
                    best_map = val_metrics['mean_metrics']['mAP']
                    self._save_checkpoint(model, optimizer, epoch, model_type, val_metrics)
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        self.logger.info(f"Early stopping triggered for {model_type}")
                        break
                
                self.logger.info(
                    f"Epoch {epoch + 1} Summary:\n"
                    f"Train Loss: {train_loss:.4f}\n"
                    f"Val mAP: {val_metrics['mean_metrics']['mAP']:.4f}"
                )
            
            scheduler.step()
            dist.barrier()

    def save_training_results(self, model_type):
        """Save training results and generate publication materials."""
        if self.rank == 0:
            # Save training history
            history_path = os.path.join(
                self.output_dir,
                f'{model_type}_training_history.json'
            )
            with open(history_path, 'w') as f:
                json.dump(self.training_history[model_type], f, indent=4)
            
            # Generate comprehensive report
            self.metrics_analyzer.generate_metrics_report(
                self.evaluation_metrics[model_type],
                self.training_history[model_type],
                [model_type]
            )
            
            self.logger.info(f"Saved training results for {model_type}")

    def finalize_training(self):
        """Generate final metrics report comparing all models."""
        if self.rank == 0:
            # Generate comparative analysis
            self.metrics_analyzer.generate_metrics_report(
                [self.evaluation_metrics[m_type] for m_type in self.model_configs.keys()],
                [self.training_history[m_type] for m_type in self.model_configs.keys()],
                list(self.model_configs.keys())
            )
            
            self.logger.info("Generated final metrics report")

    def _save_checkpoint(self, model, optimizer, epoch, model_type, metrics):
        """Save model checkpoint."""
        if self.rank == 0:
            checkpoint_dir = os.path.join(self.output_dir, "checkpoints", model_type)
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.module.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': metrics
            }
            
            path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
            torch.save(checkpoint, path)
            
            self.logger.info(f"Saved checkpoint for {model_type} epoch {epoch}")

    def save_models(self, model_dir):
        """Save final models and metrics to model directory."""
        if self.rank == 0:
            os.makedirs(model_dir, exist_ok=True)
            
            # Save metrics summary
            metrics_summary = {
                model_type: metrics['final_metrics']
                for model_type, metrics in self.evaluation_metrics.items()
            }
            
            with open(os.path.join(model_dir, 'metrics_summary.json'), 'w') as f:
                json.dump(metrics_summary, f, indent=4)
            
            self.logger.info(f"Saved final models and metrics to {model_dir}")
