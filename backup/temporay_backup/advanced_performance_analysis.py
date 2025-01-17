import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os


class AdvancedPerformanceAnalyzer:
    def __init__(self, metrics_path):
        """
        Advanced performance analyzer with statistical rigor
        """
        with open(metrics_path, 'r') as f:
            self.metrics_data = json.load(f)

        self.test_cases = {
            'bright_light': 'Bright Light',
            'dim': 'Dim Conditions',
            'fog': 'Foggy',
            'night': 'Night',
            'rain': 'Rainy'
        }

    def statistical_analysis(self):
        """
        Perform comprehensive statistical analysis
        """
        results = {}
        for test_case, condition_name in self.test_cases.items():
            pytorch_times = self._extract_inference_times(test_case, 'pytorch')
            tensorflow_times = self._extract_inference_times(test_case, 'tensorflow')

            # T-test to compare means
            t_statistic, p_value = stats.ttest_ind(pytorch_times, tensorflow_times)

            results[condition_name] = {
                'pytorch': {
                    'mean': np.mean(pytorch_times),
                    'std': np.std(pytorch_times),
                    'median': np.median(pytorch_times)
                },
                'tensorflow': {
                    'mean': np.mean(tensorflow_times),
                    'std': np.std(tensorflow_times),
                    'median': np.median(tensorflow_times)
                },
                'statistical_test': {
                    't_statistic': t_statistic,
                    'p_value': p_value
                }
            }

        return results

    def _extract_inference_times(self, test_case, model_name):
        """
        Extract and filter inference times
        """
        perf_metrics = self.metrics_data['performance_metrics'][test_case][model_name]

        inference_times = [
            time for time in perf_metrics.get('inference_times', [])
            if time is not None and -1000 < time < 1000
        ]

        return inference_times

    def generate_comprehensive_report(self, output_dir):
        """
        Generate a detailed scientific report
        """
        os.makedirs(output_dir, exist_ok=True)

        # Statistical analysis
        statistical_results = self.statistical_analysis()

        # Visualization: Boxplot of Inference Times
        plt.figure(figsize=(12, 6))
        boxplot_data = {
            condition: {
                'PyTorch': statistical_results[condition]['pytorch']['mean'],
                'TensorFlow': statistical_results[condition]['tensorflow']['mean']
            } for condition in statistical_results
        }

        conditions = list(boxplot_data.keys())
        pytorch_means = [boxplot_data[cond]['PyTorch'] for cond in conditions]
        tensorflow_means = [boxplot_data[cond]['TensorFlow'] for cond in conditions]

        x = np.arange(len(conditions))
        width = 0.35

        plt.bar(x - width / 2, pytorch_means, width, label='PyTorch', color='blue', alpha=0.7)
        plt.bar(x + width / 2, tensorflow_means, width, label='TensorFlow', color='green', alpha=0.7)

        plt.xlabel('Environmental Conditions')
        plt.ylabel('Average Inference Time (ms)')
        plt.title('Inference Time Comparison with Statistical Significance')
        plt.xticks(x, conditions, rotation=45)
        plt.legend()
        plt.tight_layout()

        plt.savefig(os.path.join(output_dir, 'inference_times_statistical.png'))
        plt.close()

        # Generate detailed report
        report_path = os.path.join(output_dir, 'statistical_performance_report.txt')
        with open(report_path, 'w') as f:
            f.write("Comprehensive Statistical Performance Analysis\n")
            f.write("=" * 50 + "\n\n")

            for condition, results in statistical_results.items():
                f.write(f"Condition: {condition}\n")
                f.write("-" * 20 + "\n")

                f.write("PyTorch Model:\n")
                f.write(f"  Mean Inference Time: {results['pytorch']['mean']:.2f} ms\n")
                f.write(f"  Standard Deviation: {results['pytorch']['std']:.2f} ms\n")
                f.write(f"  Median Inference Time: {results['pytorch']['median']:.2f} ms\n\n")

                f.write("TensorFlow Model:\n")
                f.write(f"  Mean Inference Time: {results['tensorflow']['mean']:.2f} ms\n")
                f.write(f"  Standard Deviation: {results['tensorflow']['std']:.2f} ms\n")
                f.write(f"  Median Inference Time: {results['tensorflow']['median']:.2f} ms\n\n")

                f.write("Statistical Significance:\n")
                f.write(f"  T-Statistic: {results['statistical_test']['t_statistic']:.4f}\n")
                f.write(f"  P-Value: {results['statistical_test']['p_value']:.4f}\n\n")

        return statistical_results, report_path


def main():
    # Path to the metrics JSON file
    metrics_path = "/mnt/c/Users/ashut/MachineLearning/SelfAdaptiveObjectDetection/ObjectDetection_Edge/Results/test_metrics.json"

    # Output directory for analysis results
    output_dir = "/mnt/c/Users/ashut/MachineLearning/SelfAdaptiveObjectDetection/ObjectDetection_Edge/advanced_performance_analysis"

    # Create analyzer
    analyzer = AdvancedPerformanceAnalyzer(metrics_path)

    # Generate comprehensive report
    statistical_results, report_path = analyzer.generate_comprehensive_report(output_dir)

    # Print key findings
    print("Advanced Performance Analysis Complete!")
    print(f"Detailed report saved to: {report_path}")


if __name__ == "__main__":
    main()