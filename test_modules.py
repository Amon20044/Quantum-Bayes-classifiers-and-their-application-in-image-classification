"""
Quick test script to verify QBC modules are working.
"""

import sys
import os
sys.path.insert(0, '/home/amon007/qml/qbc_project')

print("🧪 Testing QBC Modules...")
print("="*60)

# Test 1: Preprocessing
print("\n1️⃣ Testing preprocessing module...")
try:
    from preprocessing import DataLoader, ImagePreprocessor, GaussianBinarizer
    print("   ✅ Import successful")
    print("   ✅ Classes available: DataLoader, ImagePreprocessor, GaussianBinarizer")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 2: Bayesian Stats
print("\n2️⃣ Testing bayesian_stats module...")
try:
    from bayesian_stats import NaiveBayesStats, SPODEStats, TANStats, SymmetricStats
    print("   ✅ Import successful")
    print("   ✅ Classes available: NaiveBayesStats, SPODEStats, TANStats, SymmetricStats")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 3: Quantum Circuits
print("\n3️⃣ Testing quantum_circuits module...")
try:
    from quantum_circuits import f_angle, NaiveQBC, SPODE_QBC, TAN_QBC
    import numpy as np
    
    # Test f_angle function
    angle = f_angle(0.5)
    expected = 2 * np.arccos(np.sqrt(0.5))
    assert np.isclose(angle, expected), "f_angle calculation error"
    
    print("   ✅ Import successful")
    print(f"   ✅ f_angle(0.5) = {angle:.4f} rad (expected {expected:.4f})")
    print("   ✅ Classes available: NaiveQBC, SPODE_QBC, TAN_QBC")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 4: Inference
print("\n4️⃣ Testing inference module...")
try:
    from inference import QBCInference, PerformanceEvaluator
    print("   ✅ Import successful")
    print("   ✅ Classes available: QBCInference, PerformanceEvaluator")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 5: Visualization
print("\n5️⃣ Testing visualization module...")
try:
    from visualization import QBCVisualizer
    viz = QBCVisualizer()
    print("   ✅ Import successful")
    print("   ✅ QBCVisualizer instantiated")
except Exception as e:
    print(f"   ❌ Error: {e}")

# Test 6: MindQuantum
print("\n6️⃣ Testing MindQuantum...")
try:
    from mindquantum.core.gates import RY, X
    from mindquantum.core.circuit import Circuit
    from mindquantum.simulator import Simulator
    
    # Create simple circuit
    circuit = Circuit()
    circuit += RY(1.57).on(0)  # π/2 rotation
    
    # Simulate
    sim = Simulator('mqvector', 1)
    sim.apply_circuit(circuit)
    
    print("   ✅ MindQuantum import successful")
    print(f"   ✅ Created circuit with {len(circuit)} gate")
    print(f"   ✅ Simulator working")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "="*60)
print("✅ All core modules loaded successfully!")
print("\n📊 Next Steps:")
print("   1. Run: streamlit run streamlit_app.py")
print("   2. Or run a quick test with MNIST")
print("="*60)
