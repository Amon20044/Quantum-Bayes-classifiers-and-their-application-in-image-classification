# 🔬 Quantum Bayesian Classifier (QBC) - Complete Implementation

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![MindQuantum](https://img.shields.io/badge/MindQuantum-0.9%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

**Modular, production-grade implementation of Quantum Bayesian Classifiers using MindQuantum**

Based on the research paper:
> **"Quantum Bayes classifiers and their application in image classification"**  
> Ming-Ming Wang & Xiao-Ying Zhang  
> *arXiv:2401.01588v2 [quant-ph]*

---

## 🌟 Features

### 🧠 Four QBC Architectures
- **Naive QBC**: Independence assumption - P(X|y) = ∏ P(xᵢ|y)
- **SPODE QBC**: Super-Parent One-Dependent Estimator with central attribute as super-parent
- **TAN QBC**: Tree-Augmented Naive Bayes using maximum spanning tree
- **Symmetric QBC**: Exploits symmetric relationships in image features

### 📊 Complete Pipeline
1. **Data Loading**: MNIST & Fashion-MNIST support
2. **Local Feature Sampling**: 3×3 grid with 7×7 blocks (9 attributes)
3. **Gaussian Binarization**: MLE-based intersection method (Eqs. 19-21)
4. **Bayesian Statistics**: Automated computation of P(y) and P(xᵢ|parents)
5. **Quantum Circuits**: Amplitude encoding with f(P) = 2·arccos(√P)
6. **Inference**: Statevector simulation with probability extraction
7. **Evaluation**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

### 🎨 Beautiful Visualizations
- Sample images & labels
- Sampling grid overlay
- Binary feature heatmaps
- Gaussian distribution plots
- Confusion matrices
- Performance comparisons
- Bayesian network graphs
- Prediction examples (correct/incorrect)

### 🌐 Interactive Streamlit App
- Real-time experimentation
- Dataset selection (MNIST/Fashion-MNIST)
- Binary classification configurator
- Multiple QBC comparison
- Comprehensive result dashboard
- Publication-quality plots

---

## 📦 Installation

### Prerequisites
- Python 3.9 or higher
- pip or conda

### Step 1: Clone Repository
```bash
cd ~/qml
# Project already exists in qbc_project/
```

### Step 2: Create Virtual Environment (Optional but Recommended)
```bash
python3 -m venv qbc_env
source qbc_env/bin/activate  # On Windows: qbc_env\Scripts\activate
```

### Step 3: Install Dependencies
```bash
cd qbc_project
pip install -r requirements.txt
```

**Note**: If MindQuantum installation fails, try:
```bash
pip install mindquantum --index-url https://pypi.org/simple
```

---

## 🚀 Quick Start

### Option 1: Streamlit Web App (Recommended)
```bash
cd qbc_project
streamlit run streamlit_app.py
```

Then open your browser at `http://localhost:8501`

### Option 2: Python Script

```python
from preprocessing import DataLoader, ImagePreprocessor
from bayesian_stats import NaiveBayesStats
from quantum_circuits import build_qbc_circuit
from inference import QBCInference, PerformanceEvaluator

# 1. Load data
loader = DataLoader('mnist_784')
X, y = loader.load_data()

# 2. Preprocess
preprocessor = ImagePreprocessor(block_size=7, grid_size=3)
X_train_binary, y_train = preprocessor.fit_transform(X[:1000], y[:1000], class0=0, class1=1)
X_test_binary = preprocessor.transform(X[1000:1200])
y_test = y[1000:1200]

# 3. Compute statistics (Naive Bayes)
stats = NaiveBayesStats()
stats.fit(X_train_binary, y_train, [0, 1])

# 4. Build quantum circuit
circuit, builder = build_qbc_circuit('naive', n_attributes=9, statistics=stats)
print(f"Circuit: {len(circuit)} gates, depth {circuit.depth()}")

# 5. Inference
inference = QBCInference(circuit, n_attributes=9)
predictions, probabilities = inference.predict(X_test_binary)

# 6. Evaluate
evaluator = PerformanceEvaluator()
metrics = evaluator.compute_metrics(y_test, predictions, class0=0, class1=1)
evaluator.print_metrics(metrics, 0, 1)
```

---

## 📂 Project Structure

```
qbc_project/
├── preprocessing.py          # Data loading, sampling, binarization
├── bayesian_stats.py         # Bayesian statistics for all structures
├── quantum_circuits.py       # MindQuantum circuit builders
├── inference.py              # Prediction & evaluation
├── visualization.py          # Matplotlib & Plotly visualizations
├── streamlit_app.py          # Interactive web interface
├── requirements.txt          # Python dependencies
├── __init__.py              # Package initialization
└── README.md                # This file
```

---

## 🧩 Module Descriptions

### 📥 `preprocessing.py`
**Classes:**
- `DataLoader`: Fetch MNIST/Fashion-MNIST from OpenML
- `LocalFeatureSampler`: Extract 3×3 grid of 7×7 blocks (9 attributes)
- `GaussianBinarizer`: MLE-based binarization using Gaussian intersection
- `ImagePreprocessor`: Complete pipeline (sampling → binarization)

**Key Methods:**
- `_find_intersections()`: Solve Gaussian intersection quadratic equation
- `_binarize_value()`: Apply Eqs. 19-21 from paper
- `fit_transform()`: One-step preprocessing

---

### 📊 `bayesian_stats.py`
**Classes:**
- `NaiveBayesStats`: Compute P(y) and P(xᵢ|y)
- `SPODEStats`: Compute P(y), P(x_super|y), P(xᵢ|y, x_super)
- `TANStats`: Build maximum spanning tree using CMI, compute P(xᵢ|y, x_parent)
- `SymmetricStats`: Use symmetric pairs for structure

**Key Methods:**
- `compute_prior()`: P(y) with Laplace smoothing
- `_compute_conditional_mutual_info()`: I(xᵢ, xⱼ|y) for TAN
- `_build_maximum_spanning_tree()`: Prim's algorithm
- `get_probabilities()`: Return fitted statistics

---

### ⚛️ `quantum_circuits.py`
**Classes:**
- `NaiveQBC`: Build Naive quantum circuit
- `SPODE_QBC`: Build SPODE circuit with 2-control gates
- `TAN_QBC`: Build TAN circuit layer-by-layer
- `SymmetricQBC`: Build symmetric circuit

**Key Functions:**
- `f_angle(p)`: f(P) = 2·arccos(√P) encoding
- `_apply_controlled_ry_on_state()`: Apply CRy with specific control states
- `build_qbc_circuit()`: Factory function

**Circuit Construction:**
- Qubit 0: Label (y)
- Qubits 1-n: Attributes (x₁...xₙ)
- RY gates for probability encoding
- Controlled-RY for dependencies
- X gates for control state flipping

---

### 🔮 `inference.py`
**Classes:**
- `QBCInference`: Run circuits and extract predictions
- `PerformanceEvaluator`: Compute comprehensive metrics

**Key Methods:**
- `predict_single()`: Get P(y=0, X*) and P(y=1, X*) from statevector
- `predict()`: Batch prediction
- `compute_metrics()`: Accuracy, Precision, Recall, F1, CM
- `compare_classifiers()`: Side-by-side comparison

**Prediction Process:**
1. Apply circuit to get statevector
2. Find basis states matching test feature X*
3. Extract amplitudes → probabilities
4. Choose class with max P(y, X*)

---

### 🎨 `visualization.py`
**Class:** `QBCVisualizer`

**Methods:**
- `plot_sample_images()`: Grid of images with labels
- `plot_sampling_grid()`: Overlay sampling blocks on image
- `plot_binary_features()`: Heatmap of binary vectors
- `plot_gaussian_distributions()`: Gaussian PDFs and intersections
- `plot_confusion_matrix()`: Annotated CM with accuracy
- `plot_metrics_comparison()`: Bar chart comparing QBCs
- `plot_interactive_comparison()`: Plotly interactive chart
- `plot_prediction_examples()`: Show correct/incorrect predictions
- `plot_bayesian_network()`: Visualize network structure

**Styles:**
- Research-quality plots for papers
- Color-coded (green=correct, red=incorrect)
- Professional fonts and layouts
- Seaborn + Matplotlib + Plotly

---

## 🧪 Example Experiments

### Experiment 1: MNIST Digits 0 vs 1
```bash
streamlit run streamlit_app.py
# In sidebar:
# - Dataset: MNIST
# - Class 0: 0, Class 1: 1
# - QBC Types: All
# - Training: 500, Test: 200
# Click "Run Experiment"
```

**Expected Results:**
- Naive: ~99.2% accuracy
- SPODE: ~99.0%
- TAN: ~96.8%
- Symmetric: ~99.4%

### Experiment 2: Fashion-MNIST T-shirt vs Trouser
```bash
# In sidebar:
# - Dataset: Fashion-MNIST
# - Class 0: 0, Class 1: 1
# - QBC Types: Naive, SPODE
```

**Expected Results:**
- Naive: ~84-86%
- SPODE: ~86-89%

### Experiment 3: Compare All Architectures
```python
from preprocessing import *
from bayesian_stats import *
from quantum_circuits import *
from inference import *
from visualization import *

# Load & preprocess
loader = DataLoader('mnist_784')
X, y = loader.load_data()
preprocessor = ImagePreprocessor()
X_train_bin, y_train = preprocessor.fit_transform(X[:2000], y[:2000], 0, 1)
X_test_bin = preprocessor.transform(X[2000:2400])
y_test = y[2000:2400]

results = {}

# Naive
stats_naive = NaiveBayesStats()
stats_naive.fit(X_train_bin, y_train, [0, 1])
circuit_naive, _ = build_qbc_circuit('naive', 9, stats_naive)
inf_naive = QBCInference(circuit_naive, 9)
pred_naive, _ = inf_naive.predict(X_test_bin)
results['Naive'] = PerformanceEvaluator.compute_metrics(y_test, pred_naive, 0, 1)

# SPODE
stats_spode = SPODEStats(super_parent_idx=4)
stats_spode.fit(X_train_bin, y_train, [0, 1])
circuit_spode, _ = build_qbc_circuit('spode', 9, stats_spode, super_parent_idx=4)
inf_spode = QBCInference(circuit_spode, 9)
pred_spode, _ = inf_spode.predict(X_test_bin)
results['SPODE'] = PerformanceEvaluator.compute_metrics(y_test, pred_spode, 0, 1)

# TAN
stats_tan = TANStats()
stats_tan.fit(X_train_bin, y_train, [0, 1])
circuit_tan, _ = build_qbc_circuit('tan', 9, stats_tan)
inf_tan = QBCInference(circuit_tan, 9)
pred_tan, _ = inf_tan.predict(X_test_bin)
results['TAN'] = PerformanceEvaluator.compute_metrics(y_test, pred_tan, 0, 1)

# Compare
PerformanceEvaluator.compare_classifiers(results)

# Visualize
viz = QBCVisualizer()
fig = viz.plot_metrics_comparison(results, title="QBC Comparison: MNIST 0 vs 1")
fig.savefig('qbc_comparison.png', dpi=300, bbox_inches='tight')
```

---

## 📊 Paper Reproduction

### Table I: Overall Performance

To reproduce Table I from the paper (45 binary classification pairs):

```python
import numpy as np
from itertools import combinations

dataset = 'mnist_784'
results_all = {}

for class0, class1 in combinations(range(10), 2):
    print(f"\nClassifying {class0} vs {class1}...")
    
    # Load & preprocess
    loader = DataLoader(dataset)
    X, y = loader.load_data()
    preprocessor = ImagePreprocessor()
    X_train_bin, y_train = preprocessor.fit_transform(X[:5000], y[:5000], class0, class1)
    X_test_bin = preprocessor.transform(X[5000:6000])
    y_test = y[5000:6000]
    y_test = y_test[(y_test == class0) | (y_test == class1)]
    X_test_bin = X_test_bin[:(len(y_test))]
    
    # Train Naive QBC
    stats = NaiveBayesStats()
    stats.fit(X_train_bin, y_train, [class0, class1])
    circuit, _ = build_qbc_circuit('naive', 9, stats)
    inf = QBCInference(circuit, 9)
    pred, _ = inf.predict(X_test_bin)
    
    metrics = PerformanceEvaluator.compute_metrics(y_test, pred, class0, class1)
    results_all[f"{class0}_vs_{class1}"] = metrics['accuracy']

# Compute statistics
avg_acc = np.mean(list(results_all.values()))
variance = np.var(list(results_all.values()))
print(f"\nAverage Accuracy: {avg_acc:.4f}")
print(f"Variance: {variance:.4f}")
```

---

## 🔬 Research Notes

### Encoding Function
The paper uses **f(P) = 2·arccos(√P)** to encode probabilities into rotation angles.

**Why this works:**
- After RY(θ) on |0⟩: cos²(θ/2)|0⟩ + sin²(θ/2)|1⟩
- Want: cos²(θ/2) = P(x=0)
- Solve: θ/2 = arccos(√P) → θ = 2·arccos(√P)

### Binarization Strategy
Uses Gaussian MLE intersection method (Eqs. 19-21):
1. Fit Gaussian N(μ₀, σ₀²) and N(μ₁, σ₁²) for two classes
2. Find intersection points by solving quadratic equation
3. Apply rules based on number of intersections (0, 1, or 2)

### Circuit Depth Analysis
| QBC Type | # Qubits | # Gates | Depth |
|----------|----------|---------|-------|
| Naive    | 10       | ~37     | ~19   |
| SPODE    | 10       | ~73     | ~37   |
| TAN      | 10       | ~65-73  | ~33-37|
| Symmetric| 10       | ~50-65  | ~25-33|

---

## 🐛 Troubleshooting

### MindQuantum Import Error
```bash
pip install mindquantum==0.9.0
# Or try:
conda install mindquantum -c mindspore -c conda-forge
```

### sklearn/scipy Issues
```bash
pip install --upgrade scikit-learn scipy numpy
```

### Streamlit Port Already in Use
```bash
streamlit run streamlit_app.py --server.port 8502
```

### Memory Issues on Large Datasets
Reduce `n_train_samples` and `n_test_samples` in Streamlit sidebar.

---

## 📚 References

1. **Original Paper:**  
   Wang, M.-M., & Zhang, X.-Y. (2024). Quantum Bayes classifiers and their application in image classification. *arXiv preprint arXiv:2401.01588v2*.

2. **MindQuantum Documentation:**  
   https://www.mindspore.cn/mindquantum/docs/en/master/index.html

3. **Datasets:**
   - MNIST: http://yann.lecun.com/exdb/mnist/
   - Fashion-MNIST: https://github.com/zalandoresearch/fashion-mnist

4. **Bayesian Networks:**
   - Zhou, Z.-H. (2021). *Machine Learning*. Springer Nature.

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Support for multi-class classification (one-vs-rest)
- Real quantum device execution (IBM, IonQ)
- Circuit optimization (gate reduction)
- More datasets (CIFAR-10, ImageNet)
- Hyperparameter tuning interface

---

## 📄 License

MIT License - see LICENSE file

---

## 🙏 Acknowledgments

- Original paper authors: Ming-Ming Wang & Xiao-Ying Zhang
- MindQuantum development team
- OpenML for dataset hosting
- Streamlit for amazing web framework

---

## 📧 Contact

For questions or collaboration:
- Open an issue on GitHub
- Email: [your-email@example.com]

---

**Made with ❤️ for Quantum Machine Learning Research**
