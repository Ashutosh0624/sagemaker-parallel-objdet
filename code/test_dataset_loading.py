import os
from pathlib import Path
import logging
from torch.utils.data import DataLoader
from nhollistic_test import VOCDetectionDataset

# Configure logging to write to a file and console
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("dataset_loading.log"),  # Save logs to a file
        logging.StreamHandler()  # Print logs to the console
    ]
)

logger = logging.getLogger(__name__)



# Add collate_fn if not already in VOCDetectionDataset
def collate_fn(batch):
    return tuple(zip(*batch))


class DatasetTester:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.logger = logging.getLogger(__name__)

    def filter_missing_files(self, dataset):
        """Filter out missing files from the dataset."""
        self.logger.info("Filtering out missing files...")
        valid_image_ids = []
        missing_files = []

        for img_id in dataset.image_ids:
            image_path = self.data_dir / img_id
            if image_path.exists():
                valid_image_ids.append(img_id)
            else:
                missing_files.append(img_id)

        # Log missing files
        self.logger.warning(f"Found {len(missing_files)} missing files.")
        if missing_files:
            self.logger.warning("First 5 missing files:")
            for img_id in missing_files[:5]:
                self.logger.warning(f"  - {img_id}")

        # Update dataset with valid image IDs
        dataset.image_ids = valid_image_ids
        self.logger.info(f"Dataset filtered: {len(valid_image_ids)} valid images remaining.")
        return dataset

    def prepare_dataloader(self, dataset, batch_size: int = 4, num_workers: int = 4):
        """Prepare the DataLoader."""
        self.logger.info("Preparing DataLoader...")
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
        self.logger.info(f"DataLoader created with batch size: {batch_size}")
        return dataloader

    def verify_paths(self, dataset):
        """Verify the existence of all image paths in the dataset."""
        self.logger.info("\nVerifying dataset paths...")
        missing_files = []

        for i, img_id in enumerate(dataset.image_ids):
            if i % 100 == 0:
                self.logger.debug(f"Verifying image {i}/{len(dataset.image_ids)}: {img_id}")

            image_path = self.data_dir / img_id
            if not image_path.exists():
                missing_files.append(img_id)

        if missing_files:
            self.logger.warning(f"Found {len(missing_files)} missing files.")
            self.logger.warning("First 5 missing files:")
            for img_id in missing_files[:5]:
                self.logger.warning(f"  - {img_id}")
        else:
            self.logger.info("All files are present.")

        return missing_files

    def test_image_loading(self, dataset, img_id: str) -> bool:
        """Test loading a specific image."""
        self.logger.info(f"\nTesting image loading for: {img_id}")

        try:
            idx = dataset.image_ids.index(img_id)
            image, target = dataset[idx]

            self.logger.info(f"Successfully loaded image {img_id}")
            self.logger.info(f"Image shape: {image.shape}")
            self.logger.info(f"Number of annotations: {len(target['boxes'])}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load image {img_id}: {str(e)}")
            return False

    def test_batch_loading(self, dataset, batch_size: int = 32):
        """Test batch loading."""
        self.logger.info("\nTesting batch loading...")

        try:
            dataloader = self.prepare_dataloader(dataset, batch_size=batch_size)
            images, targets = next(iter(dataloader))

            self.logger.info(f"Successfully loaded batch of {len(images)} images.")
            self.logger.info(f"First image shape: {images[0].shape}")
            self.logger.info(f"First target: {targets[0]}")
        except Exception as e:
            self.logger.error(f"Failed to load batch: {str(e)}")

    def test_edge_cases(self, dataset):
        """Test the dataset for edge cases."""
        self.logger.info("\nTesting edge cases...")

        for img_id in dataset.image_ids[:10]:  # Test the first 10 images
            try:
                image, target = dataset[dataset.image_ids.index(img_id)]
                self.logger.info(f"Image ID: {img_id}")
                self.logger.info(f"Image shape: {image.shape}")
                self.logger.info(f"Number of bounding boxes: {len(target['boxes'])}")
            except Exception as e:
                self.logger.error(f"Error loading image {img_id}: {str(e)}")


    def test_dataset(self, data_dir: str):
        """Test the dataset with missing file handling."""
        self.logger.info(f"Testing dataset loading from: {data_dir}")

        # Create dataset
        dataset = VOCDetectionDataset(data_dir=data_dir, train=True)
        self.logger.info(f"Initial dataset size: {len(dataset.image_ids)} images.")

        # Verify paths
        missing_files = self.verify_paths(dataset)

        # Filter out missing files
        dataset = self.filter_missing_files(dataset)

        # Test individual image loading
        if dataset.image_ids:
            test_img_id = dataset.image_ids[0]
            self.test_image_loading(dataset, test_img_id)

        self.test_batch_loading(dataset, batch_size=32)

        # Test edge cases
        self.test_edge_cases(dataset)


def main():
    data_dir = "/mnt/c/Users/ashut/MachineLearning/SelfAdaptiveObjectDetection/Pascal/VOC/train"  # Update with your dataset path
    tester = DatasetTester(data_dir)
    tester.test_dataset(data_dir)


if __name__ == "__main__":
    main()

