"""
Visualization module for Quantum Bayesian Classifiers.
Beautiful plots for papers and presentations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set beautiful plot style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['legend.fontsize'] = 10


class QBCVisualizer:
    """Professional visualizations for QBC results."""
    
    def __init__(self, style: str = 'research'):
        """
        Initialize visualizer.
        
        Args:
            style: 'research' for paper-quality or 'presentation' for slides
        """
        self.style = style
        self.colors = sns.color_palette("husl", 10)
        
    def plot_sample_images(self, images: np.ndarray, labels: np.ndarray, 
                          n_samples: int = 10, title: str = "Sample Images"):
        """
        Plot sample images from dataset.
        
        Args:
            images: Image array (n, 28, 28)
            labels: Labels
            n_samples: Number of samples to show
            title: Plot title
        """
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        
        indices = np.random.choice(len(images), n_samples, replace=False)
        
        for idx, ax in enumerate(axes.flat):
            i = indices[idx]
            ax.imshow(images[i], cmap='gray', interpolation='nearest')
            ax.set_title(f'Label: {labels[i]}', fontsize=12, fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_sampling_grid(self, image: np.ndarray, sampling_positions: List[Tuple[int, int]],
                          block_size: int = 7, title: str = "Local Feature Sampling"):
        """
        Visualize sampling positions on an image.
        
        Args:
            image: Single image (28, 28)
            sampling_positions: List of (top, left) positions
            block_size: Block size
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        ax.imshow(image, cmap='gray', interpolation='nearest')
        ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Draw rectangles for sampling blocks
        for idx, (top, left) in enumerate(sampling_positions):
            rect = plt.Rectangle((left-0.5, top-0.5), block_size, block_size,
                                linewidth=2, edgecolor='red', facecolor='none', alpha=0.7)
            ax.add_patch(rect)
            
            # Add label in center
            center_x = left + block_size / 2
            center_y = top + block_size / 2
            ax.text(center_x, center_y, f'x{idx}', 
                   fontsize=10, fontweight='bold', color='red',
                   ha='center', va='center',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='red'))
        
        ax.axis('off')
        plt.tight_layout()
        return fig
    
    def plot_binary_features(self, binary_features: np.ndarray, labels: np.ndarray,
                           n_samples: int = 20, title: str = "Binary Features"):
        """
        Visualize binary feature vectors as heatmap.
        
        Args:
            binary_features: Binary features (n, n_features)
            labels: Labels
            n_samples: Number of samples to show
            title: Plot title
        """
        indices = np.random.choice(len(binary_features), min(n_samples, len(binary_features)), replace=False)
        data = binary_features[indices]
        sample_labels = labels[indices]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sns.heatmap(data, cmap='RdYlGn', cbar_kws={'label': 'Bit Value'},
                   linewidths=0.5, linecolor='gray', ax=ax, vmin=0, vmax=1)
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_xlabel('Feature Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sample Index', fontsize=12, fontweight='bold')
        
        # Add labels on y-axis
        yticks = ax.get_yticks()
        yticklabels = [f"{i}: y={sample_labels[i]}" for i in range(len(sample_labels))]
        ax.set_yticks(range(len(sample_labels)))
        ax.set_yticklabels(yticklabels, fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def plot_gaussian_distributions(self, mu0: np.ndarray, sig0: np.ndarray,
                                   mu1: np.ndarray, sig1: np.ndarray,
                                   attr_idx: int = 0, 
                                   title: str = "Gaussian Distributions for Binarization"):
        """
        Plot Gaussian distributions for two classes and their intersection.
        
        Args:
            mu0, sig0: Mean and std for class 0
            mu1, sig1: Mean and std for class 1
            attr_idx: Attribute index to visualize
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        m0, s0 = mu0[attr_idx], sig0[attr_idx]
        m1, s1 = mu1[attr_idx], sig1[attr_idx]
        
        # Generate x range
        x_min = min(m0 - 3*s0, m1 - 3*s1)
        x_max = max(m0 + 3*s0, m1 + 3*s1)
        x = np.linspace(x_min, x_max, 1000)
        
        # Gaussian PDFs
        from scipy.stats import norm
        y0 = norm.pdf(x, m0, s0)
        y1 = norm.pdf(x, m1, s1)
        
        # Plot
        ax.plot(x, y0, 'b-', linewidth=2, label=f'Class 0: μ={m0:.3f}, σ={s0:.3f}')
        ax.fill_between(x, y0, alpha=0.3, color='blue')
        
        ax.plot(x, y1, 'r-', linewidth=2, label=f'Class 1: μ={m1:.3f}, σ={s1:.3f}')
        ax.fill_between(x, y1, alpha=0.3, color='red')
        
        ax.set_title(f"{title} (Attribute {attr_idx})", fontsize=14, fontweight='bold')
        ax.set_xlabel('Feature Value', fontsize=12, fontweight='bold')
        ax.set_ylabel('Probability Density', fontsize=12, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_confusion_matrix(self, cm: np.ndarray, class0: int, class1: int,
                            title: str = "Confusion Matrix"):
        """
        Plot confusion matrix with annotations.
        
        Args:
            cm: Confusion matrix (2, 2)
            class0, class1: Class labels
            title: Plot title
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=[f'Pred {class0}', f'Pred {class1}'],
                   yticklabels=[f'True {class0}', f'True {class1}'],
                   ax=ax, cbar_kws={'label': 'Count'}, linewidths=2, linecolor='black')
        
        ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        
        # Add accuracy text
        accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
        ax.text(1.0, -0.15, f'Accuracy: {accuracy:.2%}', 
               transform=ax.transAxes, ha='right', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        return fig
    
    def plot_metrics_comparison(self, results_dict: Dict[str, Dict],
                               title: str = "QBC Performance Comparison"):
        """
        Compare metrics across different QBC types.
        
        Args:
            results_dict: {classifier_name: metrics_dict}
            title: Plot title
        """
        classifiers = list(results_dict.keys())
        metrics_names = ['accuracy', 'precision', 'recall', 'f1']
        
        # Prepare data
        data = {metric: [] for metric in metrics_names}
        for clf in classifiers:
            for metric in metrics_names:
                data[metric].append(results_dict[clf][metric])
        
        # Create bar plot
        x = np.arange(len(classifiers))
        width = 0.2
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
        for i, (metric, color) in enumerate(zip(metrics_names, colors)):
            offset = width * (i - 1.5)
            ax.bar(x + offset, data[metric], width, label=metric.capitalize(), color=color, alpha=0.8)
        
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Classifier', fontsize=13, fontweight='bold')
        ax.set_ylabel('Score', fontsize=13, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(classifiers, fontsize=11, fontweight='bold')
        ax.legend(fontsize=11, loc='lower right')
        ax.set_ylim([0, 1.1])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, metric in enumerate(metrics_names):
            offset = width * (i - 1.5)
            for j, v in enumerate(data[metric]):
                ax.text(j + offset, v + 0.02, f'{v:.3f}', 
                       ha='center', va='bottom', fontsize=8, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def plot_interactive_comparison(self, results_dict: Dict[str, Dict]) -> go.Figure:
        """
        Create interactive Plotly comparison chart.
        
        Args:
            results_dict: {classifier_name: metrics_dict}
            
        Returns:
            Plotly figure
        """
        classifiers = list(results_dict.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        
        fig = go.Figure()
        
        for metric in metrics:
            values = [results_dict[clf][metric] for clf in classifiers]
            fig.add_trace(go.Bar(
                name=metric.capitalize(),
                x=classifiers,
                y=values,
                text=[f'{v:.3f}' for v in values],
                textposition='outside',
            ))
        
        fig.update_layout(
            title='QBC Performance Comparison',
            xaxis_title='Classifier',
            yaxis_title='Score',
            barmode='group',
            height=500,
            template='plotly_white',
            font=dict(size=12),
            title_font_size=16,
        )
        
        return fig
    
    def plot_prediction_examples(self, images: np.ndarray, y_true: np.ndarray,
                                y_pred: np.ndarray, probabilities: List[Dict],
                                n_samples: int = 10,
                                title: str = "Prediction Examples"):
        """
        Show sample predictions with probabilities.
        
        Args:
            images: Original images
            y_true: True labels
            y_pred: Predicted labels
            probabilities: List of probability dicts
            n_samples: Number to show
            title: Plot title
        """
        # Ensure all arrays have consistent length
        min_length = min(len(images), len(y_true), len(y_pred), len(probabilities))
        
        # Select samples: mix of correct and incorrect
        correct_mask = (y_true[:min_length] == y_pred[:min_length])
        incorrect_indices = np.where(~correct_mask)[0]
        correct_indices = np.where(correct_mask)[0]
        
        # Get balanced sample
        n_incorrect = min(len(incorrect_indices), n_samples // 2)
        n_correct = n_samples - n_incorrect
        
        if len(incorrect_indices) > 0:
            selected_incorrect = np.random.choice(incorrect_indices, min(n_incorrect, len(incorrect_indices)), replace=False)
        else:
            selected_incorrect = np.array([], dtype=int)
        
        if len(correct_indices) > 0:
            selected_correct = np.random.choice(correct_indices, min(n_correct, len(correct_indices)), replace=False)
        else:
            selected_correct = np.array([], dtype=int)
        
        indices = np.concatenate([selected_correct, selected_incorrect])
        
        n_show = min(len(indices), n_samples)
        cols = 5
        rows = (n_show + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 3*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
        
        for idx, ax in enumerate(axes.flat):
            if idx < n_show:
                i = int(indices[idx])
                
                # Double-check bounds
                if i >= min_length:
                    ax.axis('off')
                    continue
                    
                ax.imshow(images[i], cmap='gray', interpolation='nearest')
                
                # Color based on correctness
                is_correct = (y_true[i] == y_pred[i])
                color = 'green' if is_correct else 'red'
                symbol = '✓' if is_correct else '✗'
                
                prob_0 = probabilities[i][0]
                prob_1 = probabilities[i][1]
                
                title_text = f'{symbol} True: {y_true[i]}, Pred: {y_pred[i]}\n'
                title_text += f'P(0)={prob_0:.3f}, P(1)={prob_1:.3f}'
                
                ax.set_title(title_text, fontsize=9, fontweight='bold', color=color)
            ax.axis('off')
        
        plt.tight_layout()
        return fig
    
    def plot_bayesian_network(self, structure_type: str, tree_structure: Optional[Dict] = None,
                             n_attributes: int = 9):
        """
        Visualize Bayesian network structure.
        
        Args:
            structure_type: 'naive', 'spode', 'tan', or 'symmetric'
            tree_structure: For TAN/Symmetric, dict of {child: parent}
            n_attributes: Number of attributes
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Position nodes
        # Label y at top center
        y_pos = (6, 9)
        
        # Attributes in 3x3 grid below
        attr_positions = {}
        grid_size = int(np.sqrt(n_attributes))
        for i in range(n_attributes):
            row = i // grid_size
            col = i % grid_size
            x = col * 4 + 1
            y = 6 - row * 3
            attr_positions[i] = (x, y)
        
        # Draw label node
        circle_y = plt.Circle(y_pos, 0.4, color='#3498db', alpha=0.8, zorder=10)
        ax.add_patch(circle_y)
        ax.text(y_pos[0], y_pos[1], 'y', fontsize=14, fontweight='bold', 
               color='white', ha='center', va='center', zorder=11)
        
        # Draw attribute nodes
        for i in range(n_attributes):
            pos = attr_positions[i]
            circle = plt.Circle(pos, 0.4, color='#e74c3c', alpha=0.8, zorder=10)
            ax.add_patch(circle)
            ax.text(pos[0], pos[1], f'x{i}', fontsize=11, fontweight='bold',
                   color='white', ha='center', va='center', zorder=11)
        
        # Draw edges based on structure
        if structure_type == 'naive':
            # y -> all attributes
            for i in range(n_attributes):
                ax.annotate('', xy=attr_positions[i], xytext=y_pos,
                           arrowprops=dict(arrowstyle='->', lw=2, color='gray', alpha=0.6))
        
        elif structure_type == 'spode':
            # y -> all attributes
            for i in range(n_attributes):
                ax.annotate('', xy=attr_positions[i], xytext=y_pos,
                           arrowprops=dict(arrowstyle='->', lw=2, color='gray', alpha=0.6))
            
            # Super-parent (center, index 4) -> other attributes
            super_parent_idx = 4
            for i in range(n_attributes):
                if i != super_parent_idx:
                    ax.annotate('', xy=attr_positions[i], xytext=attr_positions[super_parent_idx],
                               arrowprops=dict(arrowstyle='->', lw=2, color='orange', alpha=0.7))
        
        elif structure_type in ['tan', 'symmetric'] and tree_structure:
            # y -> all attributes
            for i in range(n_attributes):
                ax.annotate('', xy=attr_positions[i], xytext=y_pos,
                           arrowprops=dict(arrowstyle='->', lw=2, color='gray', alpha=0.6))
            
            # Tree edges
            for child, parent in tree_structure.items():
                if parent is not None and child != parent:
                    ax.annotate('', xy=attr_positions[child], xytext=attr_positions[parent],
                               arrowprops=dict(arrowstyle='->', lw=2, color='green', alpha=0.7))
        
        ax.set_xlim(-1, 11)
        ax.set_ylim(-1, 11)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'{structure_type.upper()} Bayesian Network Structure', 
                    fontsize=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        return fig
