"""
Simple Example: Quantum Bayesian Classifier
============================================

This script demonstrates the QBC workflow without requiring scipy.
Uses synthetic data for demonstration.
"""

import numpy as np
import sys
sys.path.insert(0, '/home/amon007/qml/qbc_project')

print("🔬 Quantum Bayesian Classifier - Simple Example")
print("="*70)

# Create synthetic binary features (simulating preprocessed MNIST)
np.random.seed(42)

print("\n📊 Creating synthetic dataset...")
# Class 0: features tend to be 0
# Class 1: features tend to be 1
n_train = 200
n_test = 50
n_features = 9

# Training data
X_train_class0 = (np.random.rand(n_train//2, n_features) < 0.3).astype(int)
X_train_class1 = (np.random.rand(n_train//2, n_features) < 0.7).astype(int)
X_train = np.vstack([X_train_class0, X_train_class1])
y_train = np.array([0]*(n_train//2) + [1]*(n_train//2))

# Test data  
X_test_class0 = (np.random.rand(n_test//2, n_features) < 0.3).astype(int)
X_test_class1 = (np.random.rand(n_test//2, n_features) < 0.7).astype(int)
X_test = np.vstack([X_test_class0, X_test_class1])
y_test = np.array([0]*(n_test//2) + [1]*(n_test//2))

print(f"   ✅ Training samples: {len(X_train)} (Class 0: {(y_train==0).sum()}, Class 1: {(y_train==1).sum()})")
print(f"   ✅ Test samples: {len(X_test)}")
print(f"   ✅ Features per sample: {n_features}")

# Compute simple Naive Bayes statistics manually
print("\n📈 Computing Naive Bayes statistics...")
P_y0 = (y_train == 0).mean()
P_y1 = (y_train == 1).mean()

print(f"   P(y=0) = {P_y0:.3f}")
print(f"   P(y=1) = {P_y1:.3f}")

# Compute P(xi=0 | y) for each feature
P_xi_given_y0 = np.zeros(n_features)
P_xi_given_y1 = np.zeros(n_features)

for i in range(n_features):
    # Laplace smoothing
    P_xi_given_y0[i] = (X_train[y_train==0, i] == 0).sum() + 1
    P_xi_given_y0[i] /= ((y_train==0).sum() + 2)
    
    P_xi_given_y1[i] = (X_train[y_train==1, i] == 0).sum() + 1
    P_xi_given_y1[i] /= ((y_train==1).sum() + 2)

print(f"   ✅ Computed conditional probabilities for {n_features} features")

# Encoding function
def f_angle(p):
    """f(P) = 2 * arccos(sqrt(P))"""
    p = np.clip(p, 1e-12, 1.0)
    return 2 * np.arccos(np.sqrt(p))

print(f"\n⚛️  Encoding probabilities to quantum angles...")
theta_y = f_angle(P_y0)
print(f"   Label angle: θ_y = {theta_y:.4f} rad")

theta_features_y0 = [f_angle(p) for p in P_xi_given_y0]
theta_features_y1 = [f_angle(p) for p in P_xi_given_y1]
print(f"   ✅ Computed {len(theta_features_y0)} rotation angles per class")

try:
    print("\n🔮 Building Quantum Circuit with MindQuantum...")
    from mindquantum.core.gates import RY, X
    from mindquantum.core.circuit import Circuit
    from mindquantum.simulator import Simulator
    
    # Build Naive QBC circuit
    circuit = Circuit()
    
    # Encode prior P(y=0)
    circuit += RY(theta_y).on(0)
    
    # For each attribute
    for i in range(n_features):
        qubit_idx = i + 1
        
        # Controlled RY with y=1
        circuit += RY(theta_features_y1[i]).on(qubit_idx, 0)
        
        # Controlled RY with y=0
        circuit += X.on(0)
        circuit += RY(theta_features_y0[i]).on(qubit_idx, 0)
        circuit += X.on(0)
    
    print(f"   ✅ Circuit built: {len(circuit)} gates, depth {circuit.depth()}")
    print(f"   ✅ Qubits: {n_features + 1} (1 label + {n_features} features)")
    
    # Inference on test set
    print("\n🎯 Running inference...")
    simulator = Simulator('mqvector', n_features + 1)
    
    correct = 0
    for i, (test_sample, true_label) in enumerate(zip(X_test, y_test)):
        simulator.reset()
        simulator.apply_circuit(circuit)
        state_vector = simulator.get_qs()
        
        # Extract probabilities for matching basis states
        probs = {0: 0.0, 1: 0.0}
        for y_val in [0, 1]:
            bitstring = str(y_val) + ''.join(map(str, test_sample))
            bitstring_le = bitstring[::-1]  # Little-endian
            basis_idx = int(bitstring_le, 2)
            amplitude = state_vector[basis_idx]
            probs[y_val] = abs(amplitude) ** 2
        
        # Predict
        pred = 0 if probs[0] > probs[1] else 1
        if pred == true_label:
            correct += 1
    
    accuracy = correct / len(y_test)
    
    print(f"   ✅ Predictions complete!")
    print(f"\n📊 Results:")
    print(f"   Correct: {correct}/{len(y_test)}")
    print(f"   Accuracy: {accuracy:.2%}")
    
    # Confusion matrix
    TP = sum((y_test == 1) & ([1 if probs[0] <= probs[1] else 0 for probs in 
              [{'0': 0.5, '1': 0.5} for _ in range(len(y_test))]]))
    
    print(f"\n{'='*70}")
    print(f"✅ Quantum Bayesian Classifier working successfully!")
    print(f"{'='*70}")
    
except ImportError as e:
    print(f"   ⚠️  MindQuantum not available: {e}")
    print(f"   💡 To install: pip install mindquantum")
    print(f"\n   📝 Classical Naive Bayes prediction instead...")
    
    # Classical Naive Bayes
    correct = 0
    for test_sample, true_label in zip(X_test, y_test):
        # P(y=0 | X) ∝ P(y=0) * ∏ P(xi | y=0)
        log_prob_0 = np.log(P_y0)
        log_prob_1 = np.log(P_y1)
        
        for i in range(n_features):
            if test_sample[i] == 0:
                log_prob_0 += np.log(P_xi_given_y0[i])
                log_prob_1 += np.log(P_xi_given_y1[i])
            else:
                log_prob_0 += np.log(1 - P_xi_given_y0[i])
                log_prob_1 += np.log(1 - P_xi_given_y1[i])
        
        pred = 0 if log_prob_0 > log_prob_1 else 1
        if pred == true_label:
            correct += 1
    
    accuracy = correct / len(y_test)
    print(f"   Classical Accuracy: {accuracy:.2%}")

print(f"\n🚀 Next Steps:")
print(f"   1. Fix scipy: python3 -m venv qbc_env && source qbc_env/bin/activate")
print(f"   2. Install packages: pip install -r requirements.txt")
print(f"   3. Run Streamlit: streamlit run streamlit_app.py")
print(f"   4. Or use real MNIST data with the preprocessing module")
print(f"\n{'='*70}")
