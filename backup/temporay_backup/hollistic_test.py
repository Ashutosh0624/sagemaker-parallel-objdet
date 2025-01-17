import os
import cv2
import json
import time
import threading
import numpy as np
import warnings
import logging
import torch
import torchvision
import torchvision.transforms as transforms
import tensorflow as tf
import traceback
from datetime import datetime
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('../object_detection_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress TensorFlow and GPU-related logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
tf.config.set_visible_devices([], 'GPU')
warnings.filterwarnings("ignore")

# Test configurations (car test case removed)
TEST_CASES = {
    "bright_light": {
        "video_path": "/mnt/c/Users/ashut/MachineLearning/SelfAdaptiveObjectDetection/object_detection_project/test_videos/bright_clip.mp4",
        "frames_required": 100,
        "brightness_threshold": 0.8,
        "contrast_threshold": 0.6
    },
    "dim": {
        "video_path": "/mnt/c/Users/ashut/MachineLearning/SelfAdaptiveObjectDetection/object_detection_project/test_videos/dim_clip.mp4",
        "frames_required": 100,
        "brightness_threshold": 0.3,
        "contrast_threshold": 0.3
    },
    "fog": {
        "video_path": "/mnt/c/Users/ashut/MachineLearning/SelfAdaptiveObjectDetection/object_detection_project/test_videos/fog_clip.mp4",
        "frames_required": 100,
        "brightness_threshold": 0.4,
        "contrast_threshold": 0.4
    },
    "night": {
        "video_path": "/mnt/c/Users/ashut/MachineLearning/SelfAdaptiveObjectDetection/object_detection_project/test_videos/night_clip.mp4",
        "frames_required": 100,
        "brightness_threshold": 0.2,
        "contrast_threshold": 0.2
    },
    "rain": {
        "video_path": "/mnt/c/Users/ashut/MachineLearning/SelfAdaptiveObjectDetection/object_detection_project/test_videos/rain_clip.mp4",
        "frames_required": 100,
        "brightness_threshold": 0.4,
        "contrast_threshold": 0.4
    }
}


class ObjectDetectionTest:
    def __init__(self, tf_model_dir):
        self.results_dir = Path("../Results")
        self.results_dir.mkdir(exist_ok=True)

        # Initialize models
        self.pytorch_model = self._load_pytorch_model()
        self.tf_model = self._load_tensorflow_model(str(Path(tf_model_dir)))

        self.current_test_case = None
        self.frame_metrics = {}
        self.performance_metrics = {
            test_case: {
                'pytorch': {'inference_times': [], 'detections': []},
                'tensorflow': {'inference_times': [], 'detections': []}
            } for test_case in TEST_CASES
        }

    def _load_pytorch_model(self):
        try:
            logger.info("Loading PyTorch model...")
            model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)
            model.eval()
            logger.info("PyTorch model loaded successfully!")
            return model
        except Exception as e:
            logger.error(f"Error loading PyTorch model: {e}")
            raise

    def _load_tensorflow_model(self, model_dir):
        try:
            logger.info(f"Loading TensorFlow model from {model_dir}")
            if not os.path.exists(model_dir):
                raise FileNotFoundError(f"TensorFlow model directory not found: {model_dir}")

            logger.info(f"Directory contents: {os.listdir(model_dir)}")
            model = tf.saved_model.load(model_dir)
            logger.info("TensorFlow model loaded successfully!")
            return model
        except Exception as e:
            logger.error(f"Error loading TensorFlow model: {e}")
            raise

    def run_pytorch_inference(self, frame):
        try:
            start_time = time.time()

            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((320, 320)),
                transforms.ToTensor(),
            ])
            input_tensor = transform(frame).unsqueeze(0)

            with torch.no_grad():
                outputs = self.pytorch_model(input_tensor)

            detections = self._process_pytorch_detections(outputs[0], frame.shape[:2])
            inference_time = (time.time() - start_time) * 1000

            return detections, inference_time
        except Exception as e:
            logger.error(f"Error in PyTorch inference: {str(e)}")
            raise

    def _process_pytorch_detections(self, outputs, frame_shape):
        try:
            height, width = frame_shape
            detections = []
            confidence_threshold = 0.5

            boxes = outputs['boxes'].cpu().numpy()
            scores = outputs['scores'].cpu().numpy()
            labels = outputs['labels'].cpu().numpy()

            for box, score, label in zip(boxes, scores, labels):
                if score > confidence_threshold:
                    x1, y1, x2, y2 = [int(coord) for coord in box]

                    detections.append({
                        'bbox': np.array([x1, y1, x2, y2]),
                        'score': float(score),
                        'label': int(label)
                    })

            return detections
        except Exception as e:
            logger.error(f"Error processing PyTorch detections: {str(e)}")
            raise

    def run_tensorflow_inference(self, frame):
        try:
            start_time = time.time()

            input_tensor = cv2.resize(frame, (320, 320))
            input_tensor = np.expand_dims(input_tensor, 0)

            detections = self.tf_model(input_tensor)
            processed_detections = self._process_tensorflow_detections(detections, frame.shape[:2])

            inference_time = (time.time() - start_time) * 1000
            return processed_detections, inference_time
        except Exception as e:
            logger.error(f"Error in TensorFlow inference: {str(e)}")
            raise

    def _process_tensorflow_detections(self, detections, frame_shape):
        try:
            height, width = frame_shape
            processed_detections = []
            confidence_threshold = 0.5

            boxes = detections['detection_boxes'][0].numpy()
            scores = detections['detection_scores'][0].numpy()
            classes = detections['detection_classes'][0].numpy()

            for box, score, class_id in zip(boxes, scores, classes):
                if score > confidence_threshold:
                    y1, x1, y2, x2 = box
                    x1, x2 = int(x1 * width), int(x2 * width)
                    y1, y2 = int(y1 * height), int(y2 * height)

                    processed_detections.append({
                        'bbox': np.array([x1, y1, x2, y2]),
                        'score': float(score),
                        'label': int(class_id)
                    })

            return processed_detections
        except Exception as e:
            logger.error(f"Error processing TensorFlow detections: {str(e)}")
            raise

    def visualize_detections(self, frame, detections):
        try:
            processed_frame = frame.copy()
            colors = {1: (0, 255, 0), 2: (255, 0, 0), 3: (0, 0, 255)}
            class_names = {1: 'person', 2: 'vehicle', 3: 'animal'}

            for det in detections:
                bbox = det['bbox'].astype(int)
                score = det['score']
                label = det['label']

                color = colors.get(label, (0, 255, 0))
                class_name = class_names.get(label, f'class_{label}')
                label_text = f'{class_name}: {score:.2f}'

                cv2.rectangle(processed_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                cv2.putText(processed_frame, label_text, (bbox[0], bbox[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            return processed_frame
        except Exception as e:
            logger.error(f"Error visualizing detections: {str(e)}")
            raise

    def save_test_results(self, frame, detections, model_name, frame_count, inference_time, test_case):
        try:
            # Save frame with detections
            output_dir = self.results_dir / test_case / model_name
            output_dir.mkdir(parents=True, exist_ok=True)

            processed_frame = self.visualize_detections(frame, detections)
            frame_path = output_dir / f"frame_{frame_count:04d}.jpg"
            cv2.imwrite(str(frame_path), processed_frame)

            # Save metrics
            if frame_count not in self.frame_metrics:
                self.frame_metrics[frame_count] = {}

            self.frame_metrics[frame_count][model_name] = {
                'num_detections': len(detections),
                'inference_time': inference_time,
                'timestamp': datetime.now().isoformat(),
                'test_case': test_case
            }

            # Update performance metrics
            self.performance_metrics[test_case][model_name]['inference_times'].append(inference_time)
            self.performance_metrics[test_case][model_name]['detections'].append(len(detections))

        except Exception as e:
            logger.error(f"Error saving test results: {str(e)}")
            raise

    def run_tests(self):
        logger.info("Starting test execution...")
        test_summary = {
            'start_time': datetime.now().isoformat(),
            'test_cases': {}
        }

        for test_case, config in TEST_CASES.items():
            try:
                logger.info(f"Starting test case: {test_case}")
                test_start_time = time.time()

                # Verify video file
                if not os.path.exists(config["video_path"]):
                    logger.error(f"Video file not found: {config['video_path']}")
                    continue

                cap = cv2.VideoCapture(config["video_path"])
                if not cap.isOpened():
                    logger.error(f"Cannot open video: {config['video_path']}")
                    continue

                frame_count = 0
                test_metrics = {
                    'pytorch': {'total_time': 0, 'total_detections': 0},
                    'tensorflow': {'total_time': 0, 'total_detections': 0}
                }

                while frame_count < config["frames_required"]:
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning(f"End of video reached for {test_case}")
                        break

                    # Run inference
                    pytorch_detections, pytorch_time = self.run_pytorch_inference(frame)
                    tensorflow_detections, tensorflow_time = self.run_tensorflow_inference(frame)

                    # Update metrics
                    test_metrics['pytorch']['total_time'] += pytorch_time
                    test_metrics['pytorch']['total_detections'] += len(pytorch_detections)
                    test_metrics['tensorflow']['total_time'] += tensorflow_time
                    test_metrics['tensorflow']['total_detections'] += len(tensorflow_detections)

                    # Save results
                    self.save_test_results(frame, pytorch_detections, "pytorch", frame_count, pytorch_time, test_case)
                    self.save_test_results(frame, tensorflow_detections, "tensorflow", frame_count, tensorflow_time,
                                           test_case)

                    frame_count += 1
                    if frame_count % 10 == 0:
                        logger.info(f"Processed {frame_count}/{config['frames_required']} frames for {test_case}")

                    # Show the processed frame
                    processed_frame = self.visualize_detections(frame, pytorch_detections)
                    cv2.imshow('Object Detection Test', processed_frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("Test stopped by user")
                        break

                cap.release()

                # Save test case summary
                test_duration = time.time() - test_start_time
                test_summary['test_cases'][test_case] = {
                    'frames_processed': frame_count,
                    'duration': test_duration,
                    'pytorch_metrics': test_metrics['pytorch'],
                    'tensorflow_metrics': test_metrics['tensorflow'],
                    'avg_pytorch_time': test_metrics['pytorch']['total_time'] / frame_count if frame_count > 0 else 0,
                    'avg_tensorflow_time': test_metrics['tensorflow'][
                                               'total_time'] / frame_count if frame_count > 0 else 0
                }

                logger.info(f"Completed test case: {test_case}")
                logger.info(f"Processed {frame_count} frames in {test_duration:.2f} seconds")

            except Exception as e:
                logger.error(f"Error in test case {test_case}: {str(e)}")
                traceback.print_exc()
                continue

        test_summary['end_time'] = datetime.now().isoformat()
        self.save_final_metrics(test_summary)

    def save_final_metrics(self, test_summary):
        metrics_path = self.results_dir / "test_metrics.json"
        try:
            metrics_data = {
                'test_cases': TEST_CASES,
                'frame_metrics': self.frame_metrics,
                'performance_metrics': self.performance_metrics,
                'test_summary': test_summary,
                'timestamp': datetime.now().isoformat()
            }

            with open(metrics_path, 'w') as f:
                json.dump(metrics_data, f, indent=4)
            logger.info(f"Metrics saved to {metrics_path}")

            # Log summary statistics
            for test_case, summary in test_summary['test_cases'].items():
                logger.info(f"\nTest Case: {test_case}")
                logger.info(f"Frames Processed: {summary['frames_processed']}")
                logger.info(f"PyTorch - Avg Time: {summary['avg_pytorch_time']:.2f}ms, "
                            f"Total Detections: {summary['pytorch_metrics']['total_detections']}")
                logger.info(f"TensorFlow - Avg Time: {summary['avg_tensorflow_time']:.2f}ms, "
                            f"Total Detections: {summary['tensorflow_metrics']['total_detections']}")

        except Exception as e:
            logger.error(f"Error saving metrics: {str(e)}")
            raise

def main():
    tf_model_dir = "/mnt/c/Users/ashut/MachineLearning/SelfAdaptiveObjectDetection/ObjectDetection_Edge/tensorflow_models/ssd_mobilenet/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8/saved_model"

    try:
        logger.info("Initializing Object Detection Test")
        test_runner = ObjectDetectionTest(tf_model_dir)
        test_runner.run_tests()
        logger.info("Test execution completed successfully")
    except Exception as e:
        logger.error(f"Error during test execution: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    main()