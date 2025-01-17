import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score
import pandas as pd
import json
from datetime import datetime
import torch

class PublicationMetricsAnalyzer:
    """
    A comprehensive metrics analyzer for generating publication-quality results
    for object detection models.
    """
    
    def __init__(self, output_dir):
        """
        Initialize the metrics analyzer.
        
        Args:
            output_dir (str): Directory to save all metrics and visualizations
        """
        self.output_dir = output_dir
        self.metrics_dir = os.path.join(output_dir, 'publication_metrics')
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Set publication-quality plot styles
        plt.style.use('seaborn-whitegrid')
        self.colors = sns.color_palette('deep')
        
        # Configure plot settings for publication
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 14,
            'figure.titlesize': 16,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1
        })

    def compute_iou_metrics(self, pred_boxes, true_boxes, iou_thresholds=None):
        """
        Compute comprehensive IoU-based metrics.
        
        Args:
            pred_boxes: Predicted bounding boxes (N, 4) [x1, y1, x2, y2]
            true_boxes: Ground truth boxes (M, 4) [x1, y1, x2, y2]
            iou_thresholds: List of IoU thresholds for evaluation
            
        Returns:
            Dictionary containing various metrics
        """
        if iou_thresholds is None:
            iou_thresholds = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        
        # Compute IoU matrix
        iou_matrix = self._compute_iou_matrix(pred_boxes, true_boxes)
        
        metrics = {
            'iou_matrix': iou_matrix,
            'thresholds': {},
            'mean_metrics': {}
        }
        
        # Compute metrics for each threshold
        for threshold in iou_thresholds:
            metrics['thresholds'][threshold] = self._compute_threshold_metrics(
                iou_matrix, threshold)
        
        # Compute mean metrics
        metrics['mean_metrics'] = {
            'mAP': np.mean([m['AP'] for m in metrics['thresholds'].values()]),
            'mean_precision': np.mean([m['precision'] for m in metrics['thresholds'].values()]),
            'mean_recall': np.mean([m['recall'] for m in metrics['thresholds'].values()]),
            'mean_f1': np.mean([m['f1'] for m in metrics['thresholds'].values()])
        }
        
        return metrics

    def _compute_iou_matrix(self, boxes1, boxes2):
        """Compute IoU matrix between two sets of boxes."""
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])
        rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
        
        wh = np.clip(rb - lt, 0, None)
        intersection = wh[:, :, 0] * wh[:, :, 1]
        
        union = area1[:, None] + area2 - intersection
        
        return intersection / union

    def _compute_threshold_metrics(self, iou_matrix, threshold):
        """Compute precision, recall, F1, and AP at given IoU threshold."""
        matches = iou_matrix >= threshold
        
        true_positives = np.sum(matches, axis=1)
        false_positives = len(matches) - true_positives
        false_negatives = len(matches[0]) - true_positives
        
        precision = true_positives / (true_positives + false_positives + 1e-6)
        recall = true_positives / (true_positives + false_negatives + 1e-6)
        
        f1 = 2 * (precision * recall) / (precision + recall + 1e-6)
        ap = self._compute_average_precision(precision, recall)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'AP': ap
        }

    def _compute_average_precision(self, precision, recall):
        """Compute Average Precision using 11-point interpolation."""
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recall >= t) == 0:
                p = 0
            else:
                p = np.max(precision[recall >= t])
            ap = ap + p / 11
        return ap

    def plot_metrics_over_time(self, training_history, metrics_names=None):
        """Plot training metrics over time."""
        if metrics_names is None:
            metrics_names = ['loss', 'mAP', 'f1']
        
        fig, axes = plt.subplots(len(metrics_names), 1, 
                                figsize=(12, 4*len(metrics_names)))
        if len(metrics_names) == 1:
            axes = [axes]
        
        for ax, metric_name in zip(axes, metrics_names):
            train_metric = training_history[f'train_{metric_name}']
            val_metric = training_history[f'val_{metric_name}']
            epochs = range(1, len(train_metric) + 1)
            
            ax.plot(epochs, train_metric, 'o-', 
                   label=f'Training {metric_name}', color=self.colors[0])
            ax.plot(epochs, val_metric, 'o-', 
                   label=f'Validation {metric_name}', color=self.colors[1])
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel(metric_name.upper())
            ax.set_title(f'{metric_name.upper()} Over Time')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.metrics_dir, 'metrics_over_time.pdf'))
        plt.close()

    def plot_per_class_metrics(self, per_class_metrics):
        """Generate per-class metrics visualizations."""
        metrics_data = []
        for class_id, metrics in per_class_metrics.items():
            metrics_data.append({
                'class_id': class_id,
                'mAP': metrics['mAP'],
                'precision': metrics['mean_precision'],
                'recall': metrics['mean_recall'],
                'f1': metrics['mean_f1']
            })
        df = pd.DataFrame(metrics_data)
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Per-Class Performance Metrics', fontsize=16)
        
        metrics_to_plot = [
            ('mAP', 'Mean Average Precision'),
            ('precision', 'Precision'),
            ('recall', 'Recall'),
            ('f1', 'F1 Score')
        ]
        
        for (metric, title), ax in zip(metrics_to_plot, axes.flat):
            sns.barplot(data=df, x='class_id', y=metric, ax=ax)
            ax.set_title(f'{title} by Class')
            ax.set_xlabel('Class ID')
            ax.set_ylabel(title)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.metrics_dir, 'per_class_metrics.pdf'))
        plt.close()
        
        # Save detailed metrics to CSV
        df.to_csv(os.path.join(self.metrics_dir, 'per_class_metrics.csv'), 
                 index=False)

    def generate_latex_table(self, metrics, model_names=None):
        """Generate LaTeX table with metrics."""
        if model_names is None:
            model_names = [f'Model {i+1}' for i in range(len(metrics))]
        
        latex_table = """
\\begin{table}[h]
\\centering
\\begin{tabular}{l|cccc}
\\hline
Model & mAP & Precision & Recall & F1-Score \\\\
\\hline
"""
        
        for model_name, model_metrics in zip(model_names, metrics):
            latex_table += f"{model_name} & {model_metrics['mAP']:.3f} & "
            latex_table += f"{model_metrics['mean_precision']:.3f} & "
            latex_table += f"{model_metrics['mean_recall']:.3f} & "
            latex_table += f"{model_metrics['mean_f1']:.3f} \\\\\n"
        
        latex_table += """\\hline
\\end{tabular}
\\caption{Object Detection Performance Metrics}
\\label{tab:detection_metrics}
\\end{table}
"""
        
        with open(os.path.join(self.metrics_dir, 'metrics_table.tex'), 'w') as f:
            f.write(latex_table)

    def generate_metrics_report(self, metrics, training_history, model_names=None):
        """Generate comprehensive metrics report."""
        # Plot training curves
        self.plot_metrics_over_time(training_history)
        
        # Generate per-class analysis
        self.plot_per_class_metrics(metrics)
        
        # Generate LaTeX table
        self.generate_latex_table([metrics['mean_metrics']], model_names)
        
        # Save detailed metrics
        with open(os.path.join(self.metrics_dir, 'detailed_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

    def plot_precision_recall_curves(self, precisions, recalls, model_names=None):
        """Plot precision-recall curves."""
        plt.figure(figsize=(10, 8))
        
        if model_names is None:
            model_names = [f'Model {i+1}' for i in range(len(precisions))]
        
        for i, (precision, recall, name) in enumerate(zip(precisions, recalls, model_names)):
            plt.plot(recall, precision, label=f'{name}', color=self.colors[i])
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(os.path.join(self.metrics_dir, 'precision_recall_curves.pdf'))
        plt.close()

    def save_confusion_matrix(self, pred_labels, true_labels, class_names=None):
        """Generate and save confusion matrix."""
        confusion_mat = torch.zeros(len(class_names), len(class_names))
        for p, t in zip(pred_labels, true_labels):
            confusion_mat[t, p] += 1
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(confusion_mat, annot=True, fmt='g', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        plt.savefig(os.path.join(self.metrics_dir, 'confusion_matrix.pdf'))
        plt.close()
        
        return confusion_mat