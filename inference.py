"""
Inference and evaluation module for Quantum Bayesian Classifiers.
Handles prediction and performance metrics.
"""

import numpy as np
from mindquantum.simulator import Simulator
from mindquantum.core.circuit import Circuit
from typing import List, Tuple, Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


class QBCInference:
    """Perform inference using QBC circuit."""
    
    def __init__(self, circuit: Circuit, n_attributes: int):
        """
        Initialize inference engine.
        
        Args:
            circuit: Built QBC circuit
            n_attributes: Number of attribute qubits
        """
        self.circuit = circuit
        self.n_attributes = n_attributes
        self.n_qubits = n_attributes + 1  # +1 for label qubit
        self.simulator = Simulator('mqvector', self.n_qubits)
        
    def predict_single(self, binary_feature: np.ndarray) -> Tuple[int, Dict[int, float]]:
        """
        Predict class for a single binary feature vector.
        
        Args:
            binary_feature: Binary feature vector of length n_attributes
            
        Returns:
            predicted_class: 0 or 1
            probabilities: {0: P(y=0, X), 1: P(y=1, X)}
        """
        if len(binary_feature) != self.n_attributes:
            raise ValueError(f"Expected {self.n_attributes} features, got {len(binary_feature)}")
        
        # Run circuit to get statevector
        self.simulator.reset()
        self.simulator.apply_circuit(self.circuit)
        state_vector = self.simulator.get_qs()
        
        # Extract probabilities for basis states matching X*
        # Qubit ordering: q0 (y), q1..qn (x1..xn)
        # We want P(y=0, X*) and P(y=1, X*)
        
        probabilities = {0: 0.0, 1: 0.0}
        
        for y_val in [0, 1]:
            # Build bitstring: y_val + binary_feature
            # MindQuantum uses little-endian: q0 is rightmost bit
            bitstring = str(y_val) + ''.join(map(str, binary_feature))
            # Reverse for little-endian indexing
            bitstring_le = bitstring[::-1]
            basis_idx = int(bitstring_le, 2)
            
            # Get amplitude and compute probability
            amplitude = state_vector[basis_idx]
            probability = abs(amplitude) ** 2
            probabilities[y_val] = probability
        
        # Normalize (should already sum to 1, but just in case)
        total = sum(probabilities.values())
        if total > 0:
            probabilities = {k: v/total for k, v in probabilities.items()}
        
        # Predict class with higher probability
        predicted_class = 0 if probabilities[0] > probabilities[1] else 1
        
        return predicted_class, probabilities
    
    def predict(self, binary_features: np.ndarray, verbose: bool = False) -> Tuple[np.ndarray, List[Dict]]:
        """
        Predict classes for multiple samples.
        
        Args:
            binary_features: Array of shape (n_samples, n_attributes)
            verbose: Whether to print progress
            
        Returns:
            predictions: Array of predicted classes
            all_probabilities: List of probability dictionaries
        """
        n_samples = len(binary_features)
        predictions = np.zeros(n_samples, dtype=int)
        all_probabilities = []
        
        for i, feature in enumerate(binary_features):
            pred, probs = self.predict_single(feature)
            predictions[i] = pred
            all_probabilities.append(probs)
            
            if verbose and (i + 1) % 100 == 0:
                print(f"  Processed {i+1}/{n_samples} samples...")
        
        return predictions, all_probabilities


class PerformanceEvaluator:
    """Evaluate QBC performance with comprehensive metrics."""
    
    @staticmethod
    def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                       class0: int, class1: int) -> Dict[str, float]:
        """
        Compute classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class0, class1: The two classes
            
        Returns:
            Dictionary of metrics
        """
        # Map to binary labels (0 and 1)
        y_true_binary = np.where(y_true == class0, 0, 1)
        y_pred_binary = np.where(y_pred == class0, 0, 1)
        
        metrics = {
            'accuracy': accuracy_score(y_true_binary, y_pred_binary),
            'precision': precision_score(y_true_binary, y_pred_binary, zero_division=0),
            'recall': recall_score(y_true_binary, y_pred_binary, zero_division=0),
            'f1': f1_score(y_true_binary, y_pred_binary, zero_division=0),
        }
        
        # Confusion matrix
        cm = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1])
        metrics['confusion_matrix'] = cm
        metrics['TP'] = cm[1, 1]
        metrics['TN'] = cm[0, 0]
        metrics['FP'] = cm[0, 1]
        metrics['FN'] = cm[1, 0]
        
        return metrics
    
    @staticmethod
    def print_metrics(metrics: Dict, class0: int, class1: int):
        """
        Print metrics in a formatted way.
        
        Args:
            metrics: Dictionary of metrics
            class0, class1: The two classes
        """
        print(f"\n{'='*60}")
        print(f"📊 CLASSIFICATION RESULTS: Class {class0} vs Class {class1}")
        print(f"{'='*60}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"\n  Confusion Matrix:")
        cm = metrics['confusion_matrix']
        print(f"    [[TN={cm[0,0]:4d}  FP={cm[0,1]:4d}]")
        print(f"     [FN={cm[1,0]:4d}  TP={cm[1,1]:4d}]]")
        print(f"{'='*60}\n")
    
    @staticmethod
    def compare_classifiers(results_dict: Dict[str, Dict]) -> None:
        """
        Compare multiple QBC types side by side.
        
        Args:
            results_dict: {classifier_name: metrics_dict}
        """
        print(f"\n{'='*80}")
        print(f"🔬 CLASSIFIER COMPARISON")
        print(f"{'='*80}")
        
        # Header
        print(f"{'Classifier':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1 Score':<12}")
        print(f"{'-'*80}")
        
        # Sort by accuracy
        sorted_results = sorted(results_dict.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        for name, metrics in sorted_results:
            print(f"{name:<20} {metrics['accuracy']:<12.4f} {metrics['precision']:<12.4f} "
                  f"{metrics['recall']:<12.4f} {metrics['f1']:<12.4f}")
        
        print(f"{'='*80}\n")
        
        # Find best
        best_name, best_metrics = sorted_results[0]
        print(f"🏆 Best Classifier: {best_name}")
        print(f"   Accuracy: {best_metrics['accuracy']:.4f} ({best_metrics['accuracy']*100:.2f}%)")
