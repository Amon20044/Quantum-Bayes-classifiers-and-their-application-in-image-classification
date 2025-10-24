"""
Bayesian statistics module for computing probabilities.
Handles P(y) and P(x_i | parents) for different Bayesian network structures.
"""

import numpy as np
from typing import Dict, Tuple, List
from scipy.stats import entropy


class BayesianStatistics:
    """Base class for computing Bayesian statistics."""
    
    def __init__(self, smoothing: float = 1.0):
        """
        Initialize statistics computer.
        
        Args:
            smoothing: Laplace smoothing parameter (default: 1.0)
        """
        self.smoothing = smoothing
        self.P_y = {}
        self.fitted = False
        
    def compute_prior(self, labels: np.ndarray, classes: List[int]) -> Dict[int, float]:
        """
        Compute class prior probabilities P(y).
        
        Args:
            labels: Label array
            classes: List of class labels
            
        Returns:
            Dictionary mapping class to P(y=class)
        """
        P_y = {}
        for c in classes:
            count = np.sum(labels == c)
            P_y[c] = count / len(labels)
        return P_y


class NaiveBayesStats(BayesianStatistics):
    """Statistics for Naive Bayes: P(x_i | y)."""
    
    def __init__(self, smoothing: float = 1.0):
        super().__init__(smoothing)
        self.cond_probs = {}  # {class: {attr_idx: P(x_i=0|y=class)}}
        
    def fit(self, binary_features: np.ndarray, labels: np.ndarray, classes: List[int]):
        """
        Compute P(y) and P(x_i | y) for all attributes.
        
        Args:
            binary_features: Binary feature array (n_samples, n_features)
            labels: Label array (n_samples,)
            classes: List of class labels (e.g., [0, 1])
        """
        n_features = binary_features.shape[1]
        
        # Compute prior P(y)
        self.P_y = self.compute_prior(labels, classes)
        
        # Compute conditional probabilities P(x_i = 0 | y)
        for c in classes:
            mask = (labels == c)
            class_features = binary_features[mask]
            
            # Count x_i = 0 for each attribute
            counts_zero = (class_features == 0).sum(axis=0)
            total = mask.sum()
            
            # Apply Laplace smoothing
            p_xi_0 = (counts_zero + self.smoothing) / (total + 2 * self.smoothing)
            
            self.cond_probs[c] = p_xi_0
        
        self.fitted = True
        
    def get_probabilities(self) -> Tuple[Dict, Dict]:
        """
        Get computed probabilities.
        
        Returns:
            P_y: Prior probabilities
            cond_probs: Conditional probabilities {class: array of P(x_i=0|y)}
        """
        if not self.fitted:
            raise ValueError("Must fit before getting probabilities")
        return self.P_y, self.cond_probs


class SPODEStats(BayesianStatistics):
    """Statistics for SPODE: P(x_i | y, x_super) with a super-parent."""
    
    def __init__(self, super_parent_idx: int = 4, smoothing: float = 1.0):
        """
        Initialize SPODE statistics.
        
        Args:
            super_parent_idx: Index of super-parent attribute (default: 4 for center)
            smoothing: Laplace smoothing
        """
        super().__init__(smoothing)
        self.super_parent_idx = super_parent_idx
        self.super_parent_probs = {}  # P(x_super | y)
        self.cond_probs = {}  # P(x_i | y, x_super)
        
    def fit(self, binary_features: np.ndarray, labels: np.ndarray, classes: List[int]):
        """
        Compute P(y), P(x_super | y), and P(x_i | y, x_super).
        
        Args:
            binary_features: Binary feature array
            labels: Label array
            classes: Class labels
        """
        n_features = binary_features.shape[1]
        
        # Compute prior P(y)
        self.P_y = self.compute_prior(labels, classes)
        
        # For each class
        for c in classes:
            mask = (labels == c)
            class_features = binary_features[mask]
            
            # P(x_super = 0 | y)
            super_col = class_features[:, self.super_parent_idx]
            count_super_0 = (super_col == 0).sum()
            total = mask.sum()
            p_super_0 = (count_super_0 + self.smoothing) / (total + 2 * self.smoothing)
            self.super_parent_probs[c] = p_super_0
            
            # P(x_i = 0 | y, x_super) for each attribute
            self.cond_probs[c] = {}
            
            for attr_idx in range(n_features):
                if attr_idx == self.super_parent_idx:
                    continue  # Skip super-parent itself
                
                # For each configuration of (y, x_super)
                probs_given_super = {}
                for super_val in [0, 1]:
                    # Filter where x_super = super_val
                    super_mask = (super_col == super_val)
                    if super_mask.sum() == 0:
                        # No samples, use uniform
                        probs_given_super[super_val] = 0.5
                        continue
                    
                    attr_col = class_features[super_mask, attr_idx]
                    count_attr_0 = (attr_col == 0).sum()
                    total_super = super_mask.sum()
                    
                    p_attr_0 = (count_attr_0 + self.smoothing) / (total_super + 2 * self.smoothing)
                    probs_given_super[super_val] = p_attr_0
                
                self.cond_probs[c][attr_idx] = probs_given_super
        
        self.fitted = True
        
    def get_probabilities(self) -> Tuple[Dict, Dict, Dict]:
        """
        Get computed probabilities.
        
        Returns:
            P_y: Prior probabilities
            super_parent_probs: P(x_super | y)
            cond_probs: P(x_i | y, x_super)
        """
        if not self.fitted:
            raise ValueError("Must fit before getting probabilities")
        return self.P_y, self.super_parent_probs, self.cond_probs


class TANStats(BayesianStatistics):
    """Statistics for Tree-Augmented Naive Bayes (TAN)."""
    
    def __init__(self, smoothing: float = 1.0):
        super().__init__(smoothing)
        self.tree_structure = None  # Dict: {child_idx: parent_idx or None}
        self.root_idx = None
        self.cond_probs = {}  # P(x_i | y, x_parent)
        
    def _compute_conditional_mutual_info(self, binary_features: np.ndarray, 
                                         labels: np.ndarray, 
                                         i: int, j: int) -> float:
        """
        Compute conditional mutual information I(x_i, x_j | y).
        
        From paper Eq. 17:
        I(x_i, x_j | y) = Σ P(x_i, x_j | y) * log(P(x_i, x_j | y) / (P(x_i|y) * P(x_j|y)))
        
        Args:
            binary_features: Binary features
            labels: Labels
            i, j: Attribute indices
            
        Returns:
            Conditional mutual information
        """
        cmi = 0.0
        
        for y_val in np.unique(labels):
            mask = (labels == y_val)
            if mask.sum() == 0:
                continue
            
            features_y = binary_features[mask]
            p_y = mask.mean()
            
            for xi_val in [0, 1]:
                for xj_val in [0, 1]:
                    # P(x_i=xi_val, x_j=xj_val | y)
                    mask_ij = (features_y[:, i] == xi_val) & (features_y[:, j] == xj_val)
                    p_xi_xj_given_y = (mask_ij.sum() + self.smoothing) / (mask.sum() + 4 * self.smoothing)
                    
                    # P(x_i=xi_val | y)
                    mask_i = (features_y[:, i] == xi_val)
                    p_xi_given_y = (mask_i.sum() + self.smoothing) / (mask.sum() + 2 * self.smoothing)
                    
                    # P(x_j=xj_val | y)
                    mask_j = (features_y[:, j] == xj_val)
                    p_xj_given_y = (mask_j.sum() + self.smoothing) / (mask.sum() + 2 * self.smoothing)
                    
                    if p_xi_xj_given_y > 0:
                        cmi += p_y * p_xi_xj_given_y * np.log(p_xi_xj_given_y / (p_xi_given_y * p_xj_given_y + 1e-10))
        
        return cmi
    
    def _build_maximum_spanning_tree(self, binary_features: np.ndarray, labels: np.ndarray) -> Dict:
        """
        Build maximum weighted spanning tree using Prim's algorithm.
        Weights are conditional mutual information I(x_i, x_j | y).
        
        Returns:
            tree_structure: Dict mapping child_idx to parent_idx
        """
        n_features = binary_features.shape[1]
        
        # Compute CMI matrix
        print("📊 Computing conditional mutual information matrix...")
        cmi_matrix = np.zeros((n_features, n_features))
        for i in range(n_features):
            for j in range(i+1, n_features):
                cmi = self._compute_conditional_mutual_info(binary_features, labels, i, j)
                cmi_matrix[i, j] = cmi
                cmi_matrix[j, i] = cmi
        
        # Prim's algorithm for maximum spanning tree
        in_tree = [False] * n_features
        parent = [-1] * n_features
        max_weight = [-np.inf] * n_features
        
        # Start with node 0 as root
        self.root_idx = 0
        max_weight[0] = 0
        
        for _ in range(n_features):
            # Find max weight node not in tree
            u = -1
            for i in range(n_features):
                if not in_tree[i] and (u == -1 or max_weight[i] > max_weight[u]):
                    u = i
            
            in_tree[u] = True
            
            # Update weights of adjacent nodes
            for v in range(n_features):
                if not in_tree[v] and cmi_matrix[u, v] > max_weight[v]:
                    max_weight[v] = cmi_matrix[u, v]
                    parent[v] = u
        
        # Convert to tree structure
        tree_structure = {}
        for i in range(n_features):
            tree_structure[i] = parent[i] if parent[i] != -1 else None
        
        return tree_structure
    
    def fit(self, binary_features: np.ndarray, labels: np.ndarray, classes: List[int]):
        """
        Compute TAN structure and probabilities.
        
        Args:
            binary_features: Binary features
            labels: Labels
            classes: Class labels
        """
        n_features = binary_features.shape[1]
        
        # Compute prior P(y)
        self.P_y = self.compute_prior(labels, classes)
        
        # Build tree structure
        print("🌲 Building maximum spanning tree...")
        self.tree_structure = self._build_maximum_spanning_tree(binary_features, labels)
        
        # Compute conditional probabilities
        for c in classes:
            mask = (labels == c)
            class_features = binary_features[mask]
            self.cond_probs[c] = {}
            
            for attr_idx in range(n_features):
                parent_idx = self.tree_structure[attr_idx]
                
                if parent_idx is None:
                    # Root node: P(x_i = 0 | y)
                    count_0 = (class_features[:, attr_idx] == 0).sum()
                    total = mask.sum()
                    p_0 = (count_0 + self.smoothing) / (total + 2 * self.smoothing)
                    self.cond_probs[c][attr_idx] = {None: p_0}
                else:
                    # Non-root: P(x_i = 0 | y, x_parent)
                    probs_given_parent = {}
                    parent_col = class_features[:, parent_idx]
                    
                    for parent_val in [0, 1]:
                        parent_mask = (parent_col == parent_val)
                        if parent_mask.sum() == 0:
                            probs_given_parent[parent_val] = 0.5
                            continue
                        
                        attr_col = class_features[parent_mask, attr_idx]
                        count_0 = (attr_col == 0).sum()
                        total_parent = parent_mask.sum()
                        p_0 = (count_0 + self.smoothing) / (total_parent + 2 * self.smoothing)
                        probs_given_parent[parent_val] = p_0
                    
                    self.cond_probs[c][attr_idx] = probs_given_parent
        
        self.fitted = True
        print("✅ TAN structure built successfully")
        
    def get_probabilities(self) -> Tuple[Dict, Dict, Dict]:
        """
        Get computed probabilities.
        
        Returns:
            P_y: Prior probabilities
            tree_structure: Tree structure
            cond_probs: Conditional probabilities
        """
        if not self.fitted:
            raise ValueError("Must fit before getting probabilities")
        return self.P_y, self.tree_structure, self.cond_probs


class SymmetricStats(BayesianStatistics):
    """Statistics for Symmetric Bayesian Network."""
    
    def __init__(self, symmetric_pairs: List[Tuple[int, int]] = None, smoothing: float = 1.0):
        """
        Initialize Symmetric statistics.
        
        Args:
            symmetric_pairs: List of (i, j) tuples indicating symmetric attribute pairs
            smoothing: Laplace smoothing
        """
        super().__init__(smoothing)
        self.symmetric_pairs = symmetric_pairs or []
        self.tree_structure = None
        self.cond_probs = {}
        
    def _build_symmetric_structure(self, n_features: int) -> Dict:
        """
        Build tree structure based on symmetric relationships.
        
        For a 3x3 grid (9 attributes), typical symmetries:
        - Vertical: (0,6), (1,7), (2,8)
        - Horizontal: (0,2), (3,5), (6,8)
        - Diagonal: may have (0,8), (2,6)
        
        Returns:
            tree_structure: Dict mapping child_idx to parent_idx or None
        """
        # Start with naive structure (all None)
        tree_structure = {i: None for i in range(n_features)}
        
        # Add edges for symmetric pairs
        # Each pair forms a small tree
        for i, j in self.symmetric_pairs:
            if tree_structure[j] is None:
                tree_structure[j] = i
        
        return tree_structure
    
    def fit(self, binary_features: np.ndarray, labels: np.ndarray, classes: List[int]):
        """
        Compute symmetric structure and probabilities.
        
        Args:
            binary_features: Binary features
            labels: Labels
            classes: Class labels
        """
        n_features = binary_features.shape[1]
        
        # Compute prior P(y)
        self.P_y = self.compute_prior(labels, classes)
        
        # Build symmetric structure
        self.tree_structure = self._build_symmetric_structure(n_features)
        
        # Compute conditional probabilities
        for c in classes:
            mask = (labels == c)
            class_features = binary_features[mask]
            self.cond_probs[c] = {}
            
            for attr_idx in range(n_features):
                parent_idx = self.tree_structure[attr_idx]
                
                if parent_idx is None:
                    # Independent node: P(x_i = 0 | y)
                    count_0 = (class_features[:, attr_idx] == 0).sum()
                    total = mask.sum()
                    p_0 = (count_0 + self.smoothing) / (total + 2 * self.smoothing)
                    self.cond_probs[c][attr_idx] = {None: p_0}
                else:
                    # Has parent: P(x_i = 0 | y, x_parent)
                    probs_given_parent = {}
                    parent_col = class_features[:, parent_idx]
                    
                    for parent_val in [0, 1]:
                        parent_mask = (parent_col == parent_val)
                        if parent_mask.sum() == 0:
                            probs_given_parent[parent_val] = 0.5
                            continue
                        
                        attr_col = class_features[parent_mask, attr_idx]
                        count_0 = (attr_col == 0).sum()
                        total_parent = parent_mask.sum()
                        p_0 = (count_0 + self.smoothing) / (total_parent + 2 * self.smoothing)
                        probs_given_parent[parent_val] = p_0
                    
                    self.cond_probs[c][attr_idx] = probs_given_parent
        
        self.fitted = True
        
    def get_probabilities(self) -> Tuple[Dict, Dict, Dict]:
        """Get computed probabilities."""
        if not self.fitted:
            raise ValueError("Must fit before getting probabilities")
        return self.P_y, self.tree_structure, self.cond_probs
