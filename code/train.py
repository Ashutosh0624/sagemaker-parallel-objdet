import os
import sys
import argparse
import json
import torch
import torch.distributed as dist
import logging
from enhanced_parallel_trainer import EnhancedParallelTrainer
from test_dataset_loading import DatasetTester
from nhollistic_test import VOCDetectionDataset
from metrics_analyzer import PublicationMetricsAnalyzer

def setup_logging():
    """Configure logging for the training script."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def setup_distributed_training():
    """Initialize the distributed training environment for SageMaker."""
    if dist.is_initialized():
        return
    
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank

def prepare_datasets(data_dir, batch_size, num_workers, rank, world_size):
    """
    Prepare and validate datasets using DatasetTester.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Preparing datasets from {data_dir}")
    
    # Initialize DatasetTester
    tester = DatasetTester(data_dir)
    
    # Create and test datasets
    train_dataset = VOCDetectionDataset(data_dir=data_dir, train=True)
    val_dataset = VOCDetectionDataset(data_dir=data_dir, train=False)
    
    # Filter missing files
    train_dataset = tester.filter_missing_files(train_dataset)
    val_dataset = tester.filter_missing_files(val_dataset)
    
    # Verify paths
    tester.verify_paths(train_dataset)
    tester.verify_paths(val_dataset)
    
    # Test edge cases
    if rank == 0:
        tester.test_edge_cases(train_dataset)
    
    # Create distributed samplers
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank
    )
    
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank
    )
    
    # Create dataloaders
    train_loader = tester.prepare_dataloader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=train_sampler
    )
    
    val_loader = tester.prepare_dataloader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=val_sampler
    )
    
    if rank == 0:
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Validation dataset size: {len(val_dataset)}")
        
        # Test batch loading
        tester.test_batch_loading(train_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, train_sampler, val_sampler

def parse_args():
    """Parse SageMaker training arguments."""
    parser = argparse.ArgumentParser()
    
    # SageMaker parameters
    parser.add_argument('--data-dir', type=str, 
                       default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--model-dir', type=str, 
                       default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--output-data-dir', type=str, 
                       default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    parser.add_argument('--num-gpus', type=int, 
                       default=int(os.environ.get('SM_NUM_GPUS', 1)))
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--num-classes', type=int, default=21)
    
    # Model configuration
    parser.add_argument('--model-types', type=str, nargs='+', 
                       default=['frcnn', 'retinanet'])
    
    return parser.parse_args()

def main():
    """Main training function."""
    try:
        # Parse arguments and setup environment
        args = parse_args()
        logger = setup_logging()
        
        # Setup distributed training
        rank, world_size, local_rank = setup_distributed_training()
        
        if rank == 0:
            logger.info("Starting training script")
            logger.info(f"Arguments: {args}")
            logger.info(f"Number of GPUs: {args.num_gpus}")
            logger.info(f"World size: {world_size}")
        
        # Initialize metrics analyzer if rank 0
        if rank == 0:
            metrics_analyzer = PublicationMetricsAnalyzer(args.output_data_dir)
        
        # Create model configurations
        model_configs = {
            model_type: {
                'num_classes': args.num_classes,
                'learning_rate': args.learning_rate,
                'batch_size': args.batch_size,
                'num_epochs': args.epochs
            } for model_type in args.model_types
        }
        
        # Prepare datasets
        train_loader, val_loader, train_sampler, val_sampler = prepare_datasets(
            args.data_dir,
            args.batch_size,
            args.num_workers,
            rank,
            world_size
        )
        
        # Initialize trainer
        trainer = EnhancedParallelTrainer(
            data_dir=args.data_dir,
            model_configs=model_configs,
            rank=rank,
            world_size=world_size,
            output_dir=args.output_data_dir
        )
        
        # Set dataloaders and samplers
        trainer.train_loader = train_loader
        trainer.val_loader = val_loader
        trainer.train_sampler = train_sampler
        trainer.val_sampler = val_sampler
        
        # Set metrics analyzer
        if rank == 0:
            trainer.metrics_analyzer = metrics_analyzer
        
        # Training pipeline
        trainer.train_models_parallel()
        
        # Save final results and generate metrics
        if rank == 0:
            trainer.finalize_training()
            trainer.save_models(args.model_dir)
            logger.info("Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        raise
    finally:
        # Cleanup
        if dist.is_initialized():
            dist.destroy_process_group()

if __name__ == '__main__':
    main()