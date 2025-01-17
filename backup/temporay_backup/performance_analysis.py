import json
import matplotlib.pyplot as plt
import numpy as np
import os


class ModelPerformanceAnalyzer:
    def __init__(self, metrics_path):
        """
        Initialize analyzer with metrics JSON
        """
        with open(metrics_path, 'r') as f:
            self.metrics_data = json.load(f)

        # Predefined test cases and their environmental conditions
        self.test_cases = {
            'bright_light': 'Bright Light',
            'dim': 'Dim Conditions',
            'fog': 'Foggy',
            'night': 'Night',
            'rain': 'Rainy'
        }

    def generate_comprehensive_report(self, output_dir):
        """
        Generate a comprehensive performance report
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Performance metrics extraction
        performance_summary = {}
        for test_case, condition_name in self.test_cases.items():
            performance_summary[condition_name] = {
                'PyTorch': self._extract_model_metrics(test_case, 'pytorch'),
                'TensorFlow': self._extract_model_metrics(test_case, 'tensorflow')
            }

        # Generate visualizations
        self._plot_inference_times(performance_summary, output_dir)
        self._plot_inference_variability(performance_summary, output_dir)

        # Generate text report
        report_path = os.path.join(output_dir, 'performance_report.txt')
        with open(report_path, 'w') as f:
            f.write("Comprehensive Model Performance Analysis\n")
            f.write("=" * 40 + "\n\n")

            for condition, models in performance_summary.items():
                f.write(f"Condition: {condition}\n")
                f.write("-" * 20 + "\n")
                for model_name, metrics in models.items():
                    f.write(f"{model_name} Model:\n")
                    f.write(f"  Avg Inference Time: {metrics['avg_inference_time']:.2f} ms\n")
                    f.write(f"  Inference Time Variability: {metrics['std_inference_time']:.2f} ms\n")
                f.write("\n")

        return performance_summary, report_path

    def _extract_model_metrics(self, test_case, model_name):
        """
        Extract metrics for a specific model and test case
        """
        perf_metrics = self.metrics_data['performance_metrics'][test_case][model_name]

        # Filter out extreme inference times
        inference_times = [
            time for time in perf_metrics.get('inference_times', [])
            if time is not None and -1000 < time < 1000
        ]

        return {
            'total_detections': len(perf_metrics.get('detections', [])),
            'avg_inference_time': np.mean(inference_times) if inference_times else 0,
            'std_inference_time': np.std(inference_times) if inference_times else 0,
            'inference_times': inference_times
        }

    def _plot_inference_times(self, performance_summary, output_dir):
        """
        Plot average inference times across different conditions
        """
        plt.figure(figsize=(10, 6))

        conditions = list(performance_summary.keys())
        pytorch_times = [
            performance_summary[condition]['PyTorch']['avg_inference_time']
            for condition in conditions
        ]
        tensorflow_times = [
            performance_summary[condition]['TensorFlow']['avg_inference_time']
            for condition in conditions
        ]

        x = np.arange(len(conditions))
        width = 0.35

        plt.bar(x - width / 2, pytorch_times, width, label='PyTorch', color='blue', alpha=0.7)
        plt.bar(x + width / 2, tensorflow_times, width, label='TensorFlow', color='green', alpha=0.7)

        plt.xlabel('Environmental Conditions')
        plt.ylabel('Average Inference Time (ms)')
        plt.title('Average Inference Times Across Different Conditions')
        plt.xticks(x, conditions, rotation=45)
        plt.legend()
        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, 'inference_times_comparison.png'))
        plt.close()

    def _plot_inference_variability(self, performance_summary, output_dir):
        """
        Plot inference time variability (standard deviation)
        """
        plt.figure(figsize=(10, 6))

        conditions = list(performance_summary.keys())
        pytorch_std = [
            performance_summary[condition]['PyTorch']['std_inference_time']
            for condition in conditions
        ]
        tensorflow_std = [
            performance_summary[condition]['TensorFlow']['std_inference_time']
            for condition in conditions
        ]

        x = np.arange(len(conditions))
        width = 0.35

        plt.bar(x - width / 2, pytorch_std, width, label='PyTorch', color='red', alpha=0.7)
        plt.bar(x + width / 2, tensorflow_std, width, label='TensorFlow', color='orange', alpha=0.7)

        plt.xlabel('Environmental Conditions')
        plt.ylabel('Inference Time Variability (Standard Deviation, ms)')
        plt.title('Inference Time Variability Across Different Conditions')
        plt.xticks(x, conditions, rotation=45)
        plt.legend()
        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, 'inference_variability_comparison.png'))
        plt.close()


def main():
    # Path to the metrics JSON file
    metrics_path = "/mnt/c/Users/ashut/MachineLearning/SelfAdaptiveObjectDetection/ObjectDetection_Edge/Results/test_metrics.json"

    # Output directory for analysis results
    output_dir = "/mnt/c/Users/ashut/MachineLearning/SelfAdaptiveObjectDetection/ObjectDetection_Edge/performance_analysis"

    # Create analyzer
    analyzer = ModelPerformanceAnalyzer(metrics_path)

    # Generate comprehensive report
    performance_summary, report_path = analyzer.generate_comprehensive_report(output_dir)

    # Print summary to console
    print("Performance Analysis Complete!")
    print(f"Detailed report saved to: {report_path}")
    print("\nKey Findings:")
    for condition, models in performance_summary.items():
        print(f"\n{condition}:")
        for model_name, metrics in models.items():
            print(f"  {model_name} Model:")
            print(f"    Avg Inference Time: {metrics['avg_inference_time']:.2f} ms")
            print(f"    Inference Time Variability: {metrics['std_inference_time']:.2f} ms")


if __name__ == "__main__":
    main()