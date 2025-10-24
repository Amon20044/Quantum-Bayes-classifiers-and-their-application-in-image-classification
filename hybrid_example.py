"""
Hybrid QBC Example: Combining Classical GaussianNB with Quantum Classifiers.

This demonstrates the best-practice approach combining:
1. GaussianNB for feature selection and fast continuous feature processing
2. Quantum circuits for accurate binary classification
3. Ensemble methods for robust predictions
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import DataLoader, ImagePreprocessor
from hybrid_classifier import HybridQBCClassifier, EnsembleQBCClassifier
from visualization import QBCVisualizer
import matplotlib.pyplot as plt


def run_hybrid_qbc_demo():
    """Demonstrate hybrid quantum-classical classification."""
    
    print("=" * 80)
    print("🔬 HYBRID QUANTUM-CLASSICAL BAYESIAN CLASSIFIER DEMO")
    print("=" * 80)
    print()
    
    # ===== 1. Load and Preprocess Data =====
    print("📥 Loading MNIST dataset...")
    loader = DataLoader(dataset_name='mnist_784', n_samples=1000)
    X, y = loader.load_data()
    
    # Binary classification: digit 3 vs digit 7
    class_0, class_1 = 3, 7
    mask = (y == class_0) | (y == class_1)
    X, y = X[mask], y[mask]
    
    print(f"✅ Loaded {len(X)} samples: {np.sum(y == class_0)} class {class_0}, {np.sum(y == class_1)} class {class_1}")
    print()
    
    # ===== 2. Feature Extraction =====
    print("🔧 Extracting features...")
    
    # Continuous features: Flatten images for GaussianNB
    X_continuous_full = X.reshape(len(X), -1)  # (n_samples, 784)
    
    # Binary features: Local sampling + binarization for quantum
    preprocessor = ImagePreprocessor(
        grid_size=3,
        block_size=7,
        sampling_method='uniform'
    )
    
    positions = preprocessor.get_sampling_positions(image_size=28)
    local_features = preprocessor.compute_local_attributes(X, positions)
    
    # Fit binarization
    preprocessor.fit_binarization(local_features, y, class_0, class_1)
    X_binary = preprocessor.binarize_attributes(local_features)
    
    print(f"   Continuous features: {X_continuous_full.shape[1]} dimensions")
    print(f"   Binary features: {X_binary.shape[1]} dimensions")
    print()
    
    # ===== 3. Train-Test Split =====
    from sklearn.model_selection import train_test_split
    
    indices = np.arange(len(X))
    train_idx, test_idx = train_test_split(
        indices, 
        test_size=0.3, 
        random_state=42,
        stratify=y
    )
    
    X_cont_train = X_continuous_full[train_idx]
    X_cont_test = X_continuous_full[test_idx]
    X_bin_train = X_binary[train_idx]
    X_bin_test = X_binary[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    
    print(f"📊 Train: {len(train_idx)} samples, Test: {len(test_idx)} samples")
    print()
    
    # ===== 4. Train Hybrid Classifiers =====
    print("🚀 Training Hybrid Classifiers...")
    print()
    
    # Hybrid Naive Bayes
    print("=" * 60)
    print("METHOD 1: Hybrid Naive QBC (GaussianNB + Quantum Naive)")
    print("=" * 60)
    hybrid_naive = HybridQBCClassifier(
        qbc_type='naive',
        n_features_quantum=9,
        use_feature_selection=True,
        feature_selection_method='gaussian_nb',
        classical_weight=0.3,
        quantum_weight=0.7,
        smoothing=1.0
    )
    hybrid_naive.fit(X_cont_train, X_bin_train, y_train)
    print()
    
    # Hybrid SPODE
    print("=" * 60)
    print("METHOD 2: Hybrid SPODE QBC (GaussianNB + Quantum SPODE)")
    print("=" * 60)
    hybrid_spode = HybridQBCClassifier(
        qbc_type='spode',
        n_features_quantum=9,
        use_feature_selection=True,
        feature_selection_method='mutual_info',
        classical_weight=0.25,
        quantum_weight=0.75,
        smoothing=1.0,
        super_parent_idx=4
    )
    hybrid_spode.fit(X_cont_train, X_bin_train, y_train)
    print()
    
    # Ensemble Approach
    print("=" * 60)
    print("METHOD 3: Ensemble QBC (Multiple Quantum + GaussianNB weighting)")
    print("=" * 60)
    ensemble = EnsembleQBCClassifier(
        qbc_types=['naive', 'spode', 'tan'],
        use_gaussian_weights=True,
        smoothing=1.0
    )
    ensemble.fit(X_cont_train, X_bin_train, y_train)
    print()
    
    # ===== 5. Evaluation =====
    print("=" * 80)
    print("📊 PERFORMANCE EVALUATION")
    print("=" * 80)
    print()
    
    # Hybrid Naive
    print("🔹 Hybrid Naive QBC:")
    comparison_naive = hybrid_naive.get_model_comparison(X_cont_test, X_bin_test, y_test)
    for key, value in comparison_naive.items():
        print(f"   {key:25s}: {value:.4f}")
    print()
    
    # Hybrid SPODE
    print("🔹 Hybrid SPODE QBC:")
    comparison_spode = hybrid_spode.get_model_comparison(X_cont_test, X_bin_test, y_test)
    for key, value in comparison_spode.items():
        print(f"   {key:25s}: {value:.4f}")
    print()
    
    # Ensemble
    print("🔹 Ensemble QBC:")
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    
    y_pred_ensemble = ensemble.predict(X_cont_test, X_bin_test)
    acc_ensemble = accuracy_score(y_test, y_pred_ensemble)
    f1_ensemble = f1_score(y_test, y_pred_ensemble, average='weighted')
    
    print(f"   Ensemble Accuracy        : {acc_ensemble:.4f}")
    print(f"   Ensemble F1-Score        : {f1_ensemble:.4f}")
    print()
    
    # ===== 6. Feature Importance Analysis =====
    print("=" * 80)
    print("🎯 FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)
    print()
    
    importance = hybrid_naive.get_feature_importance()
    print("🔹 Top 10 features selected by GaussianNB:")
    top_features = np.argsort(importance['classical'])[-10:][::-1]
    for i, feat_idx in enumerate(top_features, 1):
        print(f"   {i}. Feature {feat_idx:3d}: Importance = {importance['classical'][feat_idx]:.6f}")
    print()
    
    print(f"🔹 Features selected for Quantum processing:")
    print(f"   {importance['selected_for_quantum']}")
    print()
    
    # ===== 7. Visualization =====
    print("=" * 80)
    print("📊 CREATING COMPARISON VISUALIZATIONS")
    print("=" * 80)
    print()
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Accuracy Comparison
    methods = ['Classical\n(GaussianNB)', 'Quantum\n(Naive)', 'Hybrid\n(Naive)', 
               'Quantum\n(SPODE)', 'Hybrid\n(SPODE)', 'Ensemble']
    accuracies = [
        comparison_naive['classical_accuracy'],
        comparison_naive['quantum_accuracy'],
        comparison_naive['hybrid_accuracy'],
        comparison_spode['quantum_accuracy'],
        comparison_spode['hybrid_accuracy'],
        acc_ensemble
    ]
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#e74c3c', '#2ecc71', '#9b59b6']
    bars = axes[0, 0].bar(methods, accuracies, color=colors, alpha=0.7, edgecolor='black')
    axes[0, 0].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Classification Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylim([0.8, 1.0])
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        axes[0, 0].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: F1-Score Comparison
    f1_scores = [
        comparison_naive['classical_f1'],
        comparison_naive['quantum_f1'],
        comparison_naive['hybrid_f1'],
        comparison_spode['quantum_f1'],
        comparison_spode['hybrid_f1'],
        f1_ensemble
    ]
    
    bars = axes[0, 1].bar(methods, f1_scores, color=colors, alpha=0.7, edgecolor='black')
    axes[0, 1].set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('F1-Score Comparison', fontsize=14, fontweight='bold')
    axes[0, 1].set_ylim([0.8, 1.0])
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    for bar in bars:
        height = bar.get_height()
        axes[0, 1].text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontweight='bold')
    
    # Plot 3: Feature Importance (Top 20)
    top_20_features = np.argsort(importance['classical'])[-20:][::-1]
    top_20_importance = importance['classical'][top_20_features]
    
    axes[1, 0].barh(range(20), top_20_importance, color='steelblue', alpha=0.7)
    axes[1, 0].set_yticks(range(20))
    axes[1, 0].set_yticklabels([f'F{i}' for i in top_20_features])
    axes[1, 0].set_xlabel('Importance Score', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Top 20 Features (GaussianNB Ranking)', fontsize=14, fontweight='bold')
    axes[1, 0].invert_yaxis()
    axes[1, 0].grid(axis='x', alpha=0.3)
    
    # Plot 4: Confusion Matrix for Best Model (Ensemble)
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    
    cm = confusion_matrix(y_test, y_pred_ensemble)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[class_0, class_1],
                yticklabels=[class_0, class_1],
                ax=axes[1, 1], cbar_kws={'label': 'Count'})
    axes[1, 1].set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('True Label', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Confusion Matrix (Ensemble QBC)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('hybrid_qbc_comparison.png', dpi=300, bbox_inches='tight')
    print("✅ Saved comparison plot: hybrid_qbc_comparison.png")
    plt.show()
    
    # ===== 8. Summary =====
    print()
    print("=" * 80)
    print("📋 SUMMARY")
    print("=" * 80)
    print()
    print("✨ Key Findings:")
    print()
    print(f"1. Classical GaussianNB Baseline:")
    print(f"   - Fast training and prediction")
    print(f"   - Accuracy: {comparison_naive['classical_accuracy']:.4f}")
    print(f"   - Good for feature selection and initial filtering")
    print()
    print(f"2. Pure Quantum Classifiers:")
    print(f"   - Better probability encoding")
    print(f"   - Naive Accuracy: {comparison_naive['quantum_accuracy']:.4f}")
    print(f"   - SPODE Accuracy: {comparison_spode['quantum_accuracy']:.4f}")
    print()
    print(f"3. Hybrid Approach (Best of Both):")
    print(f"   - Combines strengths of both methods")
    print(f"   - Hybrid Naive: {comparison_naive['hybrid_accuracy']:.4f}")
    print(f"   - Hybrid SPODE: {comparison_spode['hybrid_accuracy']:.4f}")
    print()
    print(f"4. Ensemble Approach:")
    print(f"   - Most robust predictions")
    print(f"   - Accuracy: {acc_ensemble:.4f}")
    print(f"   - Recommended for production use")
    print()
    print("🎯 Recommendations:")
    print("   • Use GaussianNB for fast feature selection (30-70% weight)")
    print("   • Use Quantum circuits for accurate classification (70-30% weight)")
    print("   • Ensemble multiple QBC types for best performance")
    print("   • Feature selection reduces quantum circuit complexity")
    print()
    print("=" * 80)


if __name__ == '__main__':
    run_hybrid_qbc_demo()
