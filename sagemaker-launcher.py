import os
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.debugger import Rule, rule_configs
from sagemaker.session import Session
from sagemaker.inputs import TrainingInput
import boto3
import logging

def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def upload_dataset_to_s3(local_data_path, bucket_name, s3_prefix):
    """
    Upload dataset to S3.
    
    Args:
        local_data_path (str): Local path to dataset
        bucket_name (str): S3 bucket name
        s3_prefix (str): S3 key prefix
    
    Returns:
        str: S3 URI for uploaded data
    """
    logger = logging.getLogger(__name__)
    s3_client = boto3.client('s3')
    
    # Create S3 URI
    s3_uri = f's3://{bucket_name}/{s3_prefix}'
    
    try:
        logger.info(f"Uploading data from {local_data_path} to {s3_uri}")
        for root, _, files in os.walk(local_data_path):
            for file in files:
                local_file = os.path.join(root, file)
                relative_path = os.path.relpath(local_file, local_data_path)
                s3_key = f"{s3_prefix}/{relative_path}"
                
                # Upload file
                s3_client.upload_file(local_file, bucket_name, s3_key)
        
        logger.info("Dataset upload completed successfully")
        return s3_uri
    
    except Exception as e:
        logger.error(f"Error uploading dataset: {str(e)}")
        raise

def create_sagemaker_estimator(
    role,
    instance_count=2,
    instance_type='ml.p3.8xlarge',
    hyperparameters=None
):
    """
    Create PyTorch estimator for SageMaker training.
    
    Args:
        role (str): AWS IAM role
        instance_count (int): Number of training instances
        instance_type (str): Type of training instance
        hyperparameters (dict): Training hyperparameters
    
    Returns:
        sagemaker.pytorch.PyTorch: Configured estimator
    """
    if hyperparameters is None:
        hyperparameters = {
            'batch-size': 4,
            'epochs': 50,
            'learning-rate': 0.001,
            'num-workers': 4,
            'num-classes': 21
        }
    
    # Create PyTorch estimator
    estimator = PyTorch(
        entry_point='train.py',
        source_dir='code',
        role=role,
        framework_version='2.0.1',
        py_version='py39',
        instance_count=instance_count,
        instance_type=instance_type,
        hyperparameters=hyperparameters,
        # Enable distributed training
        distribution={
            'torch_distributed': {
                'enabled': True
            }
        },
        # Add debugging and monitoring
        debugger_hook_config=True,
        rules=[
            Rule.sagemaker(rule_configs.loss_not_decreasing()),
            Rule.sagemaker(rule_configs.gpu_utilization())
        ],
        # Define metric definitions for tracking
        metric_definitions=[
            {'Name': 'train:loss', 'Regex': 'Train Loss: ([0-9\\.]+)'},
            {'Name': 'val:mAP', 'Regex': 'Validation mAP: ([0-9\\.]+)'},
            {'Name': 'val:f1', 'Regex': 'Validation F1: ([0-9\\.]+)'}
        ],
        # Set training timeout and keep-alive
        max_run=72*3600,  # 72 hours
        keep_alive_period_in_seconds=1800  # 30 minutes
    )
    
    return estimator

def launch_training_job(
    local_data_path,
    bucket_name,
    job_name=None,
    instance_count=2,
    instance_type='ml.p3.8xlarge',
    hyperparameters=None
):
    """
    Launch SageMaker training job.
    
    Args:
        local_data_path (str): Path to local dataset
        bucket_name (str): S3 bucket name
        job_name (str): Training job name
        instance_count (int): Number of training instances
        instance_type (str): Type of training instance
        hyperparameters (dict): Training hyperparameters
    
    Returns:
        sagemaker.pytorch.PyTorch: Training estimator
    """
    logger = logging.getLogger(__name__)
    
    try:
        # Initialize SageMaker session
        session = Session()
        role = sagemaker.get_execution_role()
        
        # Upload dataset to S3
        s3_prefix = 'object-detection/data'
        s3_data_path = upload_dataset_to_s3(local_data_path, bucket_name, s3_prefix)
        
        # Create training input
        train_input = TrainingInput(
            s3_data=s3_data_path,
            distribution='FullyReplicated',
            content_type='application/x-image'
        )
        
        # Create estimator
        estimator = create_sagemaker_estimator(
            role=role,
            instance_count=instance_count,
            instance_type=instance_type,
            hyperparameters=hyperparameters
        )
        
        # Start training
        estimator.fit(
            inputs={'training': train_input},
            job_name=job_name,
            wait=False  # Don't block execution
        )
        
        logger.info(f"Training job launched: {estimator.latest_training_job.name}")
        logger.info("Monitor the training job in the SageMaker console")
        
        return estimator
    
    except Exception as e:
        logger.error(f"Error launching training job: {str(e)}")
        raise

def main():
    """Launch training job with default configuration."""
    logger = setup_logging()
    
    # Configuration
    config = {
        'local_data_path': '/path/to/your/dataset',  # Update this path
        'bucket_name': 'your-bucket-name',  # Update this bucket name
        'job_name': 'object-detection-training',
        'instance_count': 2,
        'instance_type': 'ml.p3.8xlarge',
        'hyperparameters': {
            'batch-size': 4,
            'epochs': 50,
            'learning-rate': 0.001,
            'num-workers': 4,
            'num-classes': 21
        }
    }
    
    try:
        # Launch training job
        estimator = launch_training_job(
            local_data_path=config['local_data_path'],
            bucket_name=config['bucket_name'],
            job_name=config['job_name'],
            instance_count=config['instance_count'],
            instance_type=config['instance_type'],
            hyperparameters=config['hyperparameters']
        )
        
        logger.info("Successfully initiated training job")
        return estimator
    
    except Exception as e:
        logger.error(f"Failed to launch training job: {str(e)}")
        raise

if __name__ == '__main__':
    main()