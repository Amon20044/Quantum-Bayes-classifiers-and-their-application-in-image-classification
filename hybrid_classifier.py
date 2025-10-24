"""
Hybrid Quantum-Classical Bayesian Classifier.
Combines sklearn's GaussianNB with quantum binary classifiers for enhanced performance.
"""

import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from sklearn.decomposition import PCA
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from bayesian_stats import NaiveBayesStats, SPODEStats, TANStats, SymmetricStats
from quantum_circuits import build_qbc_circuit
from inference import QBCInference


class HybridQBCClassifier:
    """
    Hybrid Quantum-Classical Bayesian Classifier.
    
    Uses GaussianNB for:
    1. Feature importance ranking
    2. Continuous feature probability estimation
    3. Initial filtering/dimensionality reduction
    
    Uses Quantum Circuits for:
    4. Final classification with selected binary features
    5. Quantum advantage in probability encoding
    
    Best Practices:
    - GaussianNB: Fast, works with continuous features, good for feature selection
    - Quantum: Better probability encoding, handles complex dependencies
    """
    
    def __init__(
        self,
        qbc_type: str = 'naive',
        n_features_quantum: int = 9,
        use_feature_selection: bool = True,
        feature_selection_method: str = 'gaussian_nb',
        classical_weight: float = 0.3,
        quantum_weight: float = 0.7,
        smoothing: float = 1.0,
        **kwargs
    ):
        """
        Initialize Hybrid QBC Classifier.
        
        Args:
            qbc_type: Type of quantum classifier ('naive', 'spode', 'tan', 'symmetric')
            n_features_quantum: Number of features to use for quantum part
            use_feature_selection: Whether to use GaussianNB for feature selection
            feature_selection_method: 'gaussian_nb', 'mutual_info', or 'f_score'
            classical_weight: Weight for GaussianNB predictions (0-1)
            quantum_weight: Weight for quantum predictions (0-1)
            smoothing: Laplace smoothing parameter
            **kwargs: Additional arguments for QBC (e.g., super_parent_idx)
        """
        self.qbc_type = qbc_type
        self.n_features_quantum = n_features_quantum
        self.use_feature_selection = use_feature_selection
        self.feature_selection_method = feature_selection_method
        self.classical_weight = classical_weight
        self.quantum_weight = quantum_weight
        self.smoothing = smoothing
        self.kwargs = kwargs
        
        # Classical component
        self.gaussian_nb = GaussianNB()
        self.feature_selector = None
        self.selected_features = None
        
        # Quantum component
        self.qbc_stats = None
        self.qbc_circuit = None
        self.qbc_builder = None
        self.qbc_inference = None
        
        # Training data storage
        self.classes_ = None
        self.fitted = False
        
    def _select_features_gaussian(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> np.ndarray:
        """
        Use GaussianNB to rank features by importance.
        
        Strategy: Train GaussianNB and use feature variances + class separation
        as importance measure.
        
        Args:
            X: Feature array (n_samples, n_features)
            y: Labels (n_samples,)
            
        Returns:
            indices: Indices of selected features
        """
        # Fit GaussianNB to get per-class statistics
        temp_gnb = GaussianNB()
        temp_gnb.fit(X, y)
        
        # Compute feature importance as variance ratio between classes
        # Higher ratio = more discriminative feature
        theta = temp_gnb.theta_  # shape: (n_classes, n_features)
        sigma = temp_gnb.var_    # shape: (n_classes, n_features)
        
        # Between-class variance / within-class variance
        between_var = np.var(theta, axis=0)  # Variance of means across classes
        within_var = np.mean(sigma, axis=0)   # Average variance within classes
        
        # Avoid division by zero
        importance = between_var / (within_var + 1e-10)
        
        # Select top k features
        indices = np.argsort(importance)[-self.n_features_quantum:]
        indices = np.sort(indices)  # Keep original order
        
        return indices
    
    def _select_features_mutual_info(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> np.ndarray:
        """
        Use mutual information for feature selection.
        
        Args:
            X: Feature array
            y: Labels
            
        Returns:
            indices: Selected feature indices
        """
        selector = SelectKBest(mutual_info_classif, k=self.n_features_quantum)
        selector.fit(X, y)
        indices = selector.get_support(indices=True)
        return indices
    
    def _select_features_f_score(
        self, 
        X: np.ndarray, 
        y: np.ndarray
    ) -> np.ndarray:
        """
        Use F-score (ANOVA) for feature selection.
        
        Args:
            X: Feature array
            y: Labels
            
        Returns:
            indices: Selected feature indices
        """
        selector = SelectKBest(f_classif, k=self.n_features_quantum)
        selector.fit(X, y)
        indices = selector.get_support(indices=True)
        return indices
    
    def fit(
        self, 
        X_continuous: np.ndarray,
        X_binary: np.ndarray, 
        y: np.ndarray
    ):
        """
        Fit hybrid classifier.
        
        Args:
            X_continuous: Continuous features for GaussianNB (n_samples, n_features)
            X_binary: Binary features for quantum part (n_samples, n_features_binary)
            y: Labels (n_samples,)
        """
        self.classes_ = np.unique(y)
        
        # === 1. Train Classical GaussianNB ===
        print("🔬 Training Classical GaussianNB...")
        self.gaussian_nb.fit(X_continuous, y)
        print(f"   ✅ GaussianNB trained on {X_continuous.shape[1]} continuous features")
        
        # === 2. Feature Selection (if enabled) ===
        if self.use_feature_selection:
            print(f"🎯 Selecting top {self.n_features_quantum} features using {self.feature_selection_method}...")
            
            if self.feature_selection_method == 'gaussian_nb':
                self.selected_features = self._select_features_gaussian(X_continuous, y)
            elif self.feature_selection_method == 'mutual_info':
                self.selected_features = self._select_features_mutual_info(X_continuous, y)
            elif self.feature_selection_method == 'f_score':
                self.selected_features = self._select_features_f_score(X_continuous, y)
            else:
                raise ValueError(f"Unknown feature selection method: {self.feature_selection_method}")
            
            print(f"   ✅ Selected features: {self.selected_features}")
            
            # Use only selected features for quantum part
            X_binary_selected = X_binary[:, self.selected_features]
        else:
            # Use all binary features
            X_binary_selected = X_binary
            self.selected_features = np.arange(X_binary.shape[1])
        
        # === 3. Train Quantum Component ===
        print(f"⚛️  Training Quantum {self.qbc_type.upper()} Classifier...")
        
        # Initialize appropriate statistics class
        if self.qbc_type == 'naive':
            self.qbc_stats = NaiveBayesStats(smoothing=self.smoothing)
        elif self.qbc_type == 'spode':
            super_parent = self.kwargs.get('super_parent_idx', 4)
            self.qbc_stats = SPODEStats(super_parent_idx=super_parent, smoothing=self.smoothing)
        elif self.qbc_type == 'tan':
            self.qbc_stats = TANStats(smoothing=self.smoothing)
        elif self.qbc_type == 'symmetric':
            self.qbc_stats = SymmetricStats(smoothing=self.smoothing)
        else:
            raise ValueError(f"Unknown QBC type: {self.qbc_type}")
        
        # Fit statistics
        self.qbc_stats.fit(X_binary_selected, y, list(self.classes_))
        
        # Build quantum circuit
        self.qbc_circuit, self.qbc_builder = build_qbc_circuit(
            qbc_type=self.qbc_type,
            n_attributes=X_binary_selected.shape[1],
            statistics=self.qbc_stats,
            **self.kwargs
        )
        
        # Create inference engine
        self.qbc_inference = QBCInference(
            circuit=self.qbc_circuit,
            builder=self.qbc_builder,
            classes=list(self.classes_)
        )
        
        print(f"   ✅ Quantum circuit built: {len(self.qbc_circuit)} gates, depth {self.qbc_circuit.depth()}")
        
        self.fitted = True
        print("✅ Hybrid QBC Classifier trained successfully!")
        
    def predict_proba(
        self, 
        X_continuous: np.ndarray,
        X_binary: np.ndarray
    ) -> np.ndarray:
        """
        Predict class probabilities using hybrid approach.
        
        Strategy:
        1. Get probabilities from GaussianNB (fast, continuous features)
        2. Get probabilities from Quantum circuit (accurate, binary features)
        3. Combine using weighted average
        
        Args:
            X_continuous: Continuous features (n_samples, n_features)
            X_binary: Binary features (n_samples, n_features_binary)
            
        Returns:
            proba: Class probabilities (n_samples, n_classes)
        """
        if not self.fitted:
            raise ValueError("Must fit before prediction")
        
        # Classical probabilities
        proba_classical = self.gaussian_nb.predict_proba(X_continuous)
        
        # Select features for quantum part
        if self.use_feature_selection:
            X_binary_selected = X_binary[:, self.selected_features]
        else:
            X_binary_selected = X_binary
        
        # Quantum probabilities
        proba_quantum = self.qbc_inference.predict_proba(X_binary_selected)
        
        # Weighted combination
        proba_hybrid = (
            self.classical_weight * proba_classical + 
            self.quantum_weight * proba_quantum
        )
        
        # Normalize to ensure sum = 1
        proba_hybrid = proba_hybrid / proba_hybrid.sum(axis=1, keepdims=True)
        
        return proba_hybrid
    
    def predict(
        self, 
        X_continuous: np.ndarray,
        X_binary: np.ndarray
    ) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X_continuous: Continuous features
            X_binary: Binary features
            
        Returns:
            predictions: Predicted class labels
        """
        proba = self.predict_proba(X_continuous, X_binary)
        return self.classes_[np.argmax(proba, axis=1)]
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """
        Get feature importance from both classical and quantum components.
        
        Returns:
            importance_dict: Dictionary with importance scores
        """
        if not self.fitted:
            raise ValueError("Must fit before getting feature importance")
        
        importance = {}
        
        # Classical feature importance (variance ratio)
        theta = self.gaussian_nb.theta_
        sigma = self.gaussian_nb.var_
        between_var = np.var(theta, axis=0)
        within_var = np.mean(sigma, axis=0)
        importance['classical'] = between_var / (within_var + 1e-10)
        
        # Selected features for quantum
        importance['selected_for_quantum'] = self.selected_features
        
        return importance
    
    def get_model_comparison(
        self,
        X_continuous: np.ndarray,
        X_binary: np.ndarray,
        y_true: np.ndarray
    ) -> Dict[str, float]:
        """
        Compare classical, quantum, and hybrid predictions.
        
        Args:
            X_continuous: Continuous features
            X_binary: Binary features
            y_true: True labels
            
        Returns:
            comparison: Dictionary with accuracy scores
        """
        from sklearn.metrics import accuracy_score, f1_score
        
        # Classical only
        y_pred_classical = self.gaussian_nb.predict(X_continuous)
        
        # Quantum only
        if self.use_feature_selection:
            X_binary_selected = X_binary[:, self.selected_features]
        else:
            X_binary_selected = X_binary
        y_pred_quantum = self.qbc_inference.predict(X_binary_selected)
        
        # Hybrid
        y_pred_hybrid = self.predict(X_continuous, X_binary)
        
        comparison = {
            'classical_accuracy': accuracy_score(y_true, y_pred_classical),
            'classical_f1': f1_score(y_true, y_pred_classical, average='weighted'),
            'quantum_accuracy': accuracy_score(y_true, y_pred_quantum),
            'quantum_f1': f1_score(y_true, y_pred_quantum, average='weighted'),
            'hybrid_accuracy': accuracy_score(y_true, y_pred_hybrid),
            'hybrid_f1': f1_score(y_true, y_pred_hybrid, average='weighted'),
        }
        
        return comparison


class EnsembleQBCClassifier:
    """
    Ensemble of multiple Quantum Bayesian Classifiers with classical voting.
    
    Strategy: Train multiple QBC types and use GaussianNB for weighted voting.
    """
    
    def __init__(
        self,
        qbc_types: List[str] = ['naive', 'spode', 'tan'],
        use_gaussian_weights: bool = True,
        smoothing: float = 1.0
    ):
        """
        Initialize Ensemble QBC.
        
        Args:
            qbc_types: List of QBC types to ensemble
            use_gaussian_weights: Use GaussianNB confidence for weighting
            smoothing: Laplace smoothing
        """
        self.qbc_types = qbc_types
        self.use_gaussian_weights = use_gaussian_weights
        self.smoothing = smoothing
        
        self.gaussian_nb = GaussianNB()
        self.qbc_classifiers = {}
        self.classes_ = None
        self.fitted = False
        
    def fit(
        self,
        X_continuous: np.ndarray,
        X_binary: np.ndarray,
        y: np.ndarray
    ):
        """
        Fit ensemble of classifiers.
        
        Args:
            X_continuous: Continuous features
            X_binary: Binary features  
            y: Labels
        """
        self.classes_ = np.unique(y)
        
        # Train GaussianNB for weighting
        if self.use_gaussian_weights:
            print("🔬 Training GaussianNB for ensemble weights...")
            self.gaussian_nb.fit(X_continuous, y)
        
        # Train each QBC type
        print(f"⚛️  Training {len(self.qbc_types)} quantum classifiers...")
        for qbc_type in self.qbc_types:
            print(f"   📊 Training {qbc_type.upper()}...")
            
            # Initialize statistics
            if qbc_type == 'naive':
                stats = NaiveBayesStats(smoothing=self.smoothing)
            elif qbc_type == 'spode':
                stats = SPODEStats(smoothing=self.smoothing)
            elif qbc_type == 'tan':
                stats = TANStats(smoothing=self.smoothing)
            elif qbc_type == 'symmetric':
                stats = SymmetricStats(smoothing=self.smoothing)
            else:
                continue
            
            # Fit and build circuit
            stats.fit(X_binary, y, list(self.classes_))
            circuit, builder = build_qbc_circuit(
                qbc_type=qbc_type,
                n_attributes=X_binary.shape[1],
                statistics=stats
            )
            
            inference = QBCInference(
                circuit=circuit,
                builder=builder,
                classes=list(self.classes_)
            )
            
            self.qbc_classifiers[qbc_type] = {
                'stats': stats,
                'circuit': circuit,
                'builder': builder,
                'inference': inference
            }
            
            print(f"      ✅ {qbc_type.upper()}: {len(circuit)} gates")
        
        self.fitted = True
        print("✅ Ensemble QBC trained successfully!")
        
    def predict_proba(
        self,
        X_continuous: np.ndarray,
        X_binary: np.ndarray
    ) -> np.ndarray:
        """
        Predict using weighted ensemble.
        
        Args:
            X_continuous: Continuous features
            X_binary: Binary features
            
        Returns:
            proba: Ensemble class probabilities
        """
        if not self.fitted:
            raise ValueError("Must fit before prediction")
        
        # Get GaussianNB confidence if using weights
        if self.use_gaussian_weights:
            gaussian_proba = self.gaussian_nb.predict_proba(X_continuous)
            gaussian_confidence = np.max(gaussian_proba, axis=1, keepdims=True)
        else:
            gaussian_confidence = np.ones((X_binary.shape[0], 1))
        
        # Collect predictions from all QBCs
        all_probas = []
        for qbc_type, classifier in self.qbc_classifiers.items():
            proba = classifier['inference'].predict_proba(X_binary)
            all_probas.append(proba)
        
        # Average with Gaussian weighting
        ensemble_proba = np.mean(all_probas, axis=0)
        
        # Apply Gaussian confidence weighting
        ensemble_proba = ensemble_proba * gaussian_confidence
        
        # Normalize
        ensemble_proba = ensemble_proba / ensemble_proba.sum(axis=1, keepdims=True)
        
        return ensemble_proba
    
    def predict(
        self,
        X_continuous: np.ndarray,
        X_binary: np.ndarray
    ) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X_continuous: Continuous features
            X_binary: Binary features
            
        Returns:
            predictions: Predicted labels
        """
        proba = self.predict_proba(X_continuous, X_binary)
        return self.classes_[np.argmax(proba, axis=1)]
