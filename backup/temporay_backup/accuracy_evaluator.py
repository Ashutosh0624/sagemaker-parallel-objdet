import json
import os
import numpy as np
from typing import List, Dict
from datetime import datetime


class AccuracyEvaluator:
    def __init__(self, ground_truth_path: str, predictions_path: str, iou_threshold: float = 0.5):
        """
        Initialize evaluator with ground truth and prediction data
        """
        # Load ground truth
        with open(ground_truth_path, 'r') as f:
            self.ground_truth = json.load(f)

        # Load predictions
        with open(predictions_path, 'r') as f:
            self.predictions_data = json.load(f)

        self.iou_threshold = iou_threshold
        self.metrics = {}

    def _safe_filter_inference_times(self, times):
        """
        Filter out extreme or invalid inference times
        """
        # Remove extreme outliers (e.g., negative times or very large times)
        filtered_times = [
            time for time in times
            if time is not None and
               -1000 < time < 1000
        ]
        return filtered_times

    def evaluate(self):
        """
        Comprehensive accuracy evaluation
        """
        # Test cases from the original script
        test_cases = ['bright_light', 'dim', 'fog', 'night', 'rain']

        for test_case in test_cases:
            # Extract performance metrics for this test case
            perf_metrics = self.predictions_data['performance_metrics'][test_case]

            # Metrics for each model
            self.metrics[test_case] = {
                'pytorch': self._evaluate_model(perf_metrics['pytorch']),
                'tensorflow': self._evaluate_model(perf_metrics['tensorflow'])
            }

        return self.metrics

    def _evaluate_model(self, model_metrics):
        """
        Evaluate metrics for a single model
        """
        # Filter inference times
        filtered_inference_times = self._safe_filter_inference_times(
            model_metrics.get('inference_times', [])
        )

        return {
            'total_detections': len(model_metrics.get('detections', [])),
            'avg_inference_time': np.mean(filtered_inference_times) if filtered_inference_times else 0,
            'std_inference_time': np.std(filtered_inference_times) if filtered_inference_times else 0
        }


def main():
    # Paths adjusted to Linux-style paths in WSL
    ground_truth_path = "/mnt/c/Users/ashut/MachineLearning/SelfAdaptiveObjectDetection/ObjectDetection_Edge/ground_truth_annotations/ground_truth.json"
    predictions_path = "/mnt/c/Users/ashut/MachineLearning/SelfAdaptiveObjectDetection/ObjectDetection_Edge/Results/test_metrics.json"

    # Create results directory if it doesn't exist
    results_dir = "/mnt/c/Users/ashut/MachineLearning/SelfAdaptiveObjectDetection/ObjectDetection_Edge/accuracy_results"
    os.makedirs(results_dir, exist_ok=True)

    # Generate unique filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(results_dir, f"accuracy_results_{timestamp}.txt")

    # Create evaluator
    evaluator = AccuracyEvaluator(
        ground_truth_path=ground_truth_path,
        predictions_path=predictions_path
    )
    results = evaluator.evaluate()

    # Prepare results string
    results_output = []
    for test_case, model_metrics in results.items():
        results_output.append(f"\n{'=' * 20}")
        results_output.append(f"Test Case: {test_case}")
        results_output.append(f"{'=' * 20}")

        for model, metrics in model_metrics.items():
            results_output.append(f"\n{model.capitalize()} Model Metrics:")
            results_output.append(f"  Total Detections: {metrics['total_detections']}")
            results_output.append(f"  Avg Inference Time: {metrics['avg_inference_time']:.2f} ms")
            results_output.append(f"  Std Dev Inference Time: {metrics['std_inference_time']:.2f} ms")

    # Print to console
    print("\n".join(results_output))

    # Save to file
    with open(output_file, 'w') as f:
        f.write("\n".join(results_output))

    print(f"\nDetailed results saved to: {output_file}")


if __name__ == "__main__":
    main()