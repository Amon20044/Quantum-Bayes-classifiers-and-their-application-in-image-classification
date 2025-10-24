"""
Quantum Bayesian Classifier (QBC) Package
==========================================

Modular implementation of Quantum Bayesian Classifiers based on the paper:
"Quantum Bayes classifiers and their application in image classification"
by Ming-Ming Wang & Xiao-Ying Zhang

Modules:
--------
- preprocessing: Data loading, sampling, and binarization
- bayesian_stats: Bayesian statistics computation for different structures
- quantum_circuits: MindQuantum circuit builders for QBCs
- inference: Prediction and evaluation
- visualization: Beautiful plots and visualizations

Usage:
------
from qbc_project.preprocessing import DataLoader, ImagePreprocessor
from qbc_project.bayesian_stats import NaiveBayesStats
from qbc_project.quantum_circuits import build_qbc_circuit
from qbc_project.inference import QBCInference
from qbc_project.visualization import QBCVisualizer
"""

__version__ = "1.0.0"
__author__ = "QBC Research Team"

__all__ = [
    'preprocessing',
    'bayesian_stats',
    'quantum_circuits',
    'inference',
    'visualization'
]
