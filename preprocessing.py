"""
Data preprocessing module for Quantum Bayesian Classifier.
Handles MNIST/Fashion-MNIST loading, local sampling, and binarization.
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from scipy.stats import norm
from typing import Tuple, List, Optional
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """Load and preprocess image datasets."""
    
    def __init__(self, dataset_name: str = 'mnist_784', n_samples: Optional[int] = None):
        """
        Initialize DataLoader.
        
        Args:
            dataset_name: 'mnist_784' or 'Fashion-MNIST'
            n_samples: Number of samples to load (None for all)
        """
        self.dataset_name = dataset_name
        self.n_samples = n_samples
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load dataset from OpenML with caching and progress feedback.
        
        Returns:
            X: Images of shape (n_samples, 28, 28)
            y: Labels of shape (n_samples,)
        """
        import os
        print(f"📥 Loading {self.dataset_name} dataset...")
        print(f"⏳ First-time download may take 30-60 seconds...")
        print(f"💾 Data will be cached in: {os.path.expanduser('~/scikit_learn_data/')}")
        
        try:
            # Use cache_dir and set timeout
            data = fetch_openml(
                self.dataset_name, 
                version=1, 
                as_frame=False, 
                parser='auto',
                data_home=None  # Uses default ~/scikit_learn_data/
            )
            print(f"✅ Dataset downloaded successfully!")
            
            X = data['data'].reshape(-1, 28, 28).astype(np.float32) / 255.0  # Normalize to [0, 1]
            y = data['target'].astype(int)
            
            if self.n_samples:
                X, y = X[:self.n_samples], y[:self.n_samples]
            
            print(f"✅ Loaded {len(X)} images of size 28x28")
            print(f"🏷️  Classes: {np.unique(y)}")
            return X, y
            
        except Exception as e:
            print(f"❌ Error loading dataset: {e}")
            print(f"💡 Tip: Check internet connection or try MNIST instead")
            raise


class LocalFeatureSampler:
    """Sample local features from images using convolution-like approach."""
    
    def __init__(self, block_size: int = 7, grid_size: int = 3):
        """
        Initialize sampler.
        
        Args:
            block_size: Size of sampling block (e.g., 7x7)
            grid_size: Number of blocks in each dimension (e.g., 3x3 = 9 blocks)
        """
        self.block_size = block_size
        self.grid_size = grid_size
        self.positions = self._get_sampling_positions()
        
    def _get_sampling_positions(self, image_size: int = 28) -> List[Tuple[int, int]]:
        """
        Calculate sampling positions for blocks.
        
        Returns:
            List of (top, left) coordinates for each block
        """
        stride = (image_size - self.block_size) // (self.grid_size - 1) if self.grid_size > 1 else 0
        positions = []
        for r in range(self.grid_size):
            for c in range(self.grid_size):
                top = r * stride
                left = c * stride
                positions.append((top, left))
        return positions
    
    def _average_pool_block(self, img: np.ndarray, top: int, left: int) -> float:
        """Average pool a single block."""
        block = img[top:top+self.block_size, left:left+self.block_size]
        return np.mean(block)
    
    def sample_features(self, images: np.ndarray) -> np.ndarray:
        """
        Sample local features from all images.
        
        Args:
            images: Array of shape (n_samples, 28, 28)
            
        Returns:
            Sampled features of shape (n_samples, n_blocks)
        """
        n_samples = len(images)
        n_blocks = len(self.positions)
        features = np.zeros((n_samples, n_blocks))
        
        for i, img in enumerate(images):
            for j, (top, left) in enumerate(self.positions):
                features[i, j] = self._average_pool_block(img, top, left)
        
        return features


class GaussianBinarizer:
    """
    Binarize continuous features using Gaussian MLE method.
    Based on the paper's equations (19)-(21).
    """
    
    def __init__(self):
        self.mu0 = None
        self.sig0 = None
        self.mu1 = None
        self.sig1 = None
        self.fitted = False
        
    def _find_intersections(self, mu0: float, sig0: float, mu1: float, sig1: float) -> List[float]:
        """
        Find intersection points of two Gaussian distributions.
        
        Solves: (1/sig0)*exp(-(x-mu0)^2/(2*sig0^2)) = (1/sig1)*exp(-(x-mu1)^2/(2*sig1^2))
        
        Returns:
            List of intersection points (0, 1, or 2 intersections)
        """
        # Coefficients for quadratic equation A*x^2 + B*x + C = 0
        A = 1/(2*sig1**2) - 1/(2*sig0**2)
        B = mu0/(sig0**2) - mu1/(sig1**2)
        C = (mu1**2)/(2*sig1**2) - (mu0**2)/(2*sig0**2) + np.log(sig1/sig0)
        
        # Linear case
        if abs(A) < 1e-12:
            if abs(B) < 1e-12:
                return []
            return [-C/B]
        
        # Quadratic case
        disc = B**2 - 4*A*C
        if disc < 0:
            return []
        if disc == 0:
            return [-B/(2*A)]
        
        sqrt_disc = np.sqrt(disc)
        r1 = (-B + sqrt_disc) / (2*A)
        r2 = (-B - sqrt_disc) / (2*A)
        return sorted([r1, r2])
    
    def _binarize_value(self, x: float, mu0: float, sig0: float, mu1: float, sig1: float) -> int:
        """
        Binarize a single value based on Gaussian intersection rules (Eqs. 19-21).
        
        Args:
            x: Value to binarize
            mu0, sig0: Parameters for class 0 Gaussian
            mu1, sig1: Parameters for class 1 Gaussian
            
        Returns:
            Binary value (0 or 1)
        """
        inters = self._find_intersections(mu0, sig0, mu1, sig1)
        
        # No intersection - use midpoint
        if len(inters) == 0:
            xins = 0.5 * (mu0 + mu1)
            if mu0 <= mu1:
                return 0 if x <= xins else 1
            else:
                return 1 if x <= xins else 0
        
        # One intersection (Eq. 19)
        if len(inters) == 1:
            xins = inters[0]
            if mu0 <= mu1:
                return 0 if x <= xins else 1
            else:
                return 1 if x <= xins else 0
        
        # Two intersections (Eqs. 20-21)
        xins1, xins2 = inters
        if mu0 <= mu1:  # Eq. 20
            if x <= xins1:
                return 0
            if xins1 <= x <= xins2:
                return 0 if abs(xins1 - x) <= abs(xins2 - x) else 1
            return 1
        else:  # Eq. 21
            if x <= xins1:
                return 1
            if xins1 <= x <= xins2:
                return 1 if abs(xins1 - x) <= abs(xins2 - x) else 0
            return 0
    
    def fit(self, features: np.ndarray, labels: np.ndarray, class0: int, class1: int):
        """
        Fit Gaussian parameters for two classes.
        
        Args:
            features: Feature array of shape (n_samples, n_features)
            labels: Label array of shape (n_samples,)
            class0, class1: The two class labels
        """
        mask0 = (labels == class0)
        mask1 = (labels == class1)
        
        self.mu0 = features[mask0].mean(axis=0)
        self.sig0 = features[mask0].std(axis=0, ddof=1) + 1e-6  # Add epsilon for stability
        self.mu1 = features[mask1].mean(axis=0)
        self.sig1 = features[mask1].std(axis=0, ddof=1) + 1e-6
        
        self.fitted = True
        
    def transform(self, features: np.ndarray) -> np.ndarray:
        """
        Transform continuous features to binary.
        
        Args:
            features: Feature array of shape (n_samples, n_features)
            
        Returns:
            Binary feature array of shape (n_samples, n_features)
        """
        if not self.fitted:
            raise ValueError("Binarizer must be fitted before transform")
        
        n_samples, n_features = features.shape
        binary_features = np.zeros_like(features, dtype=int)
        
        for i in range(n_samples):
            for j in range(n_features):
                binary_features[i, j] = self._binarize_value(
                    features[i, j],
                    self.mu0[j], self.sig0[j],
                    self.mu1[j], self.sig1[j]
                )
        
        return binary_features
    
    def fit_transform(self, features: np.ndarray, labels: np.ndarray, 
                      class0: int, class1: int) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(features, labels, class0, class1)
        return self.transform(features)


class ImagePreprocessor:
    """Complete preprocessing pipeline for QBC."""
    
    def __init__(self, block_size: int = 7, grid_size: int = 3):
        """
        Initialize preprocessor.
        
        Args:
            block_size: Size of sampling block
            grid_size: Grid size for sampling
        """
        self.sampler = LocalFeatureSampler(block_size, grid_size)
        self.binarizer = GaussianBinarizer()
        self.class0 = None
        self.class1 = None
        
    def fit(self, images: np.ndarray, labels: np.ndarray, class0: int, class1: int):
        """
        Fit the preprocessor on training data.
        
        Args:
            images: Training images
            labels: Training labels
            class0, class1: Binary classification classes
        """
        self.class0 = class0
        self.class1 = class1
        
        # Filter for binary classification
        mask = (labels == class0) | (labels == class1)
        images_binary = images[mask]
        labels_binary = labels[mask]
        
        # Sample features
        features = self.sampler.sample_features(images_binary)
        
        # Fit binarizer
        self.binarizer.fit(features, labels_binary, class0, class1)
        
    def transform(self, images: np.ndarray) -> np.ndarray:
        """
        Transform images to binary features.
        
        Args:
            images: Images to transform
            
        Returns:
            Binary features
        """
        features = self.sampler.sample_features(images)
        return self.binarizer.transform(features)
    
    def fit_transform(self, images: np.ndarray, labels: np.ndarray, 
                      class0: int, class1: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit and transform in one step.
        
        Returns:
            binary_features: Binary feature array
            binary_labels: Filtered labels for the two classes
        """
        self.fit(images, labels, class0, class1)
        
        # Filter for binary classification
        mask = (labels == class0) | (labels == class1)
        images_binary = images[mask]
        labels_binary = labels[mask]
        
        binary_features = self.transform(images_binary)
        return binary_features, labels_binary
