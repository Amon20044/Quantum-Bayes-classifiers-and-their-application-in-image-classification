# 🎯 QUICK START GUIDE - Quantum Bayesian Classifier

## ✨ What You Have

A **complete, modular Quantum Bayesian Classifier** system with:
- ✅ 4 QBC architectures (Naive, SPODE, TAN, Symmetric)
- ✅ Beautiful Streamlit web interface
- ✅ Professional visualizations
- ✅ Full MNIST/Fashion-MNIST support
- ✅ 2,800+ lines of production code
- ✅ Complete documentation

---

## 🚀 Three Ways to Run

### Option 1: Streamlit Web App (BEST FOR YOU!)

```bash
# Step 1: Create clean environment
cd ~/qml
python3 -m venv qbc_env
source qbc_env/bin/activate

# Step 2: Install packages
cd qbc_project
pip install -r requirements.txt

# Step 3: Run app
streamlit run streamlit_app.py

# Step 4: Open browser
# Visit: http://localhost:8501
```

**In the app:**
1. Select dataset (MNIST or Fashion-MNIST)
2. Choose two classes (e.g., 0 vs 1)
3. Select QBC types to compare
4. Click "Run Experiment"
5. Explore results in tabs!

---

### Option 2: Simple Python Script

```bash
cd ~/qml/qbc_project
python simple_example.py
```

This runs a synthetic example without needing scipy.

---

### Option 3: Custom Python Code

```python
# In Python terminal or Jupyter notebook
import sys
sys.path.insert(0, '/home/amon007/qml/qbc_project')

from preprocessing import DataLoader, ImagePreprocessor
from bayesian_stats import NaiveBayesStats
from quantum_circuits import build_qbc_circuit
from inference import QBCInference
from visualization import QBCVisualizer

# Load MNIST
loader = DataLoader('mnist_784')
X, y = loader.load_data()

# Preprocess (takes a few seconds)
preprocessor = ImagePreprocessor(block_size=7, grid_size=3)
X_train_bin, y_train = preprocessor.fit_transform(
    X[:1000], y[:1000], class0=0, class1=1
)
X_test_bin = preprocessor.transform(X[1000:1200])
y_test = y[1000:1200]

# Compute statistics
stats = NaiveBayesStats()
stats.fit(X_train_bin, y_train, [0, 1])

# Build circuit
circuit, builder = build_qbc_circuit('naive', 9, stats)
print(f"Circuit: {len(circuit)} gates, depth {circuit.depth()}")

# Predict
inference = QBCInference(circuit, 9)
predictions, probabilities = inference.predict(X_test_bin)

# Evaluate
from inference import PerformanceEvaluator
evaluator = PerformanceEvaluator()
metrics = evaluator.compute_metrics(y_test, predictions, 0, 1)
evaluator.print_metrics(metrics, 0, 1)

# Visualize
viz = QBCVisualizer()
fig = viz.plot_confusion_matrix(metrics['confusion_matrix'], 0, 1)
fig.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
```

---

## 📚 File Descriptions (What Each File Does)

### Core Modules

**preprocessing.py** (466 lines)
- Loads MNIST/Fashion-MNIST
- Samples 3×3 grid of 7×7 blocks (9 features)
- Binarizes using Gaussian MLE method
- `DataLoader`, `LocalFeatureSampler`, `GaussianBinarizer`, `ImagePreprocessor`

**bayesian_stats.py** (347 lines)
- Computes P(y) and P(xi|parents) for all 4 QBC types
- Implements TAN maximum spanning tree
- `NaiveBayesStats`, `SPODEStats`, `TANStats`, `SymmetricStats`

**quantum_circuits.py** (403 lines)
- Builds MindQuantum circuits
- Implements f(P) = 2·arccos(√P) encoding
- `NaiveQBC`, `SPODE_QBC`, `TAN_QBC`, `SymmetricQBC`

**inference.py** (150 lines)
- Runs quantum simulation
- Extracts predictions from statevector
- Computes accuracy, precision, recall, F1
- `QBCInference`, `PerformanceEvaluator`

**visualization.py** (524 lines)
- 10+ professional plot types
- Matplotlib + Seaborn + Plotly
- Publication-quality figures
- `QBCVisualizer`

### App & Documentation

**streamlit_app.py** (458 lines)
- Interactive web interface
- Experiment configurator
- Real-time results dashboard
- 5 tabs of visualizations

**README.md** (1000+ lines)
- Complete documentation
- Installation guide
- Usage examples
- Paper reproduction instructions

**PROJECT_SUMMARY.md**
- What we built
- How to use it
- Learning path for beginners
- Future enhancements

---

## 🎓 Learning Path (For ML Beginners)

### Week 1: Get It Running
```bash
# Day 1-2: Setup
1. Create virtual environment
2. Install dependencies
3. Run streamlit app
4. Play with different class pairs

# Day 3-5: Understand Results
1. Read confusion matrix
2. Compare Naive vs SPODE
3. Visualize sampling grid
4. Study binary features
```

### Week 2: Understand Code
```python
# Day 1-2: Preprocessing
1. Read preprocessing.py
2. Understand Gaussian binarization
3. Try different block sizes

# Day 3-5: Bayesian Networks
1. Study naive assumption
2. Understand SPODE super-parent
3. Visualize network structures
```

### Week 3: Quantum Circuits
```python
# Day 1-3: Circuit Building
1. Study f(P) encoding function
2. Understand RY gates
3. Learn controlled operations

# Day 4-5: Compare Architectures
1. Compare circuit depths
2. Analyze gate counts
3. Understand complexity
```

### Week 4: Research
```python
# Day 1-3: Reproduce Paper
1. Run all 45 class pairs
2. Compute Table I statistics
3. Compare with paper results

# Day 4-5: Experiment
1. Try Fashion-MNIST
2. Adjust hyperparameters
3. Write your own analysis
```

---

## 🔍 Key Concepts Explained

### 1. Local Feature Sampling
**Why?**
- 28×28 = 784 pixels → 784 qubits (too many!)
- Sample 9 blocks → 9 qubits (manageable)

**How?**
- Place 3×3 grid of 7×7 blocks on image
- Average pool each block
- Get 9 continuous features

### 2. Gaussian Binarization
**Why?**
- Quantum qubits are binary (|0⟩ or |1⟩)
- Need to convert continuous → binary

**How?**
- Fit Gaussian N(μ₀, σ₀²) and N(μ₁, σ₁²)
- Find intersection points
- Apply rules from paper (Eqs. 19-21)

### 3. Amplitude Encoding
**Why?**
- Want P(xi=0) encoded in amplitude

**How?**
- Use f(P) = 2·arccos(√P)
- After RY(θ): cos²(θ/2) = P
- Encodes probability in quantum state

### 4. Bayesian Networks
**Naive:**
- Assumes independence
- P(X|y) = ∏ P(xi|y)
- Simplest, fastest

**SPODE:**
- One super-parent
- P(X|y) = P(x_super|y) ∏ P(xi|y, x_super)
- More dependencies

**TAN:**
- Tree structure
- Maximum spanning tree using CMI
- Optimal tree

**Symmetric:**
- Exploits image symmetry
- Custom relationships
- Domain-specific

---

## 🐛 Troubleshooting

### Problem: scipy broken
```bash
Solution:
python3 -m venv qbc_env
source qbc_env/bin/activate
pip install -r requirements.txt
```

### Problem: MindQuantum not found
```bash
Solution:
pip install mindquantum --upgrade
```

### Problem: Streamlit port in use
```bash
Solution:
streamlit run streamlit_app.py --server.port 8502
```

### Problem: Out of memory
```bash
Solution:
Reduce samples in sidebar:
- Training: 100-200
- Test: 50-100
```

---

## 💡 Best Practices

### For Experiments
1. **Start small**: 100 training samples
2. **Use Naive first**: Get baseline
3. **Compare incrementally**: Add one QBC at a time
4. **Save results**: Screenshot or export plots
5. **Document**: Note what works

### For Code
1. **Read module first**: Understand before changing
2. **Test changes**: Run simple_example.py
3. **Use visualizer**: Plot everything
4. **Check metrics**: Don't just look at accuracy
5. **Compare**: Always have baseline

### For Research
1. **Reproduce paper**: Start with their experiments
2. **Vary one thing**: Change one parameter at a time
3. **Document everything**: Keep experiment log
4. **Visualize results**: Make plots for insights
5. **Share findings**: Contribute back

---

## 🎯 Common Use Cases

### Case 1: Quick Test
```bash
streamlit run streamlit_app.py
# Select: MNIST, 0 vs 1, Naive only, 200 train, 100 test
# Click "Run Experiment"
# Result in <30 seconds
```

### Case 2: Full Comparison
```bash
streamlit run streamlit_app.py
# Select: MNIST, 3 vs 8, All QBCs, 500 train, 200 test
# Click "Run Experiment"
# Compare in "Performance Comparison" tab
```

### Case 3: Fashion Items
```bash
streamlit run streamlit_app.py
# Select: Fashion-MNIST, 0 vs 1 (T-shirt vs Trouser)
# Select: All QBCs
# See which architecture works best for clothing
```

### Case 4: Research Paper
```python
# Run all 45 binary pairs
# Compute average accuracy
# Compare with Table I
# Write analysis
```

---

## 🌟 What Makes This Special

1. **Modular Design**
   - Each file is independent
   - Easy to understand
   - Simple to extend

2. **Beginner-Friendly**
   - Clear structure
   - Extensive comments
   - Learning path included

3. **Research-Ready**
   - Implements full paper
   - Reproducible results
   - Publication-quality plots

4. **Production-Quality**
   - Error handling
   - Input validation
   - Performance optimization

5. **Interactive**
   - Streamlit web app
   - Real-time feedback
   - Visual exploration

---

## 🎉 Success Checklist

- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] Streamlit app runs
- [ ] Experimented with MNIST 0 vs 1
- [ ] Tried all 4 QBC types
- [ ] Compared accuracies
- [ ] Explored visualizations
- [ ] Read paper sections
- [ ] Understood preprocessing
- [ ] Learned quantum encoding
- [ ] Ready to experiment!

---

## 📞 Need Help?

### Documentation
- README.md - Full guide
- PROJECT_SUMMARY.md - Overview
- Code docstrings - Inline help

### Paper
- arXiv:2401.01588v2
- Sections III & IV most relevant

### Resources
- MindQuantum docs
- Streamlit docs
- scikit-learn tutorials

---

## 🚀 You're Ready!

```bash
cd ~/qml/qbc_project
python3 -m venv qbc_env
source qbc_env/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

**Open browser → Experiment → Learn → Enjoy quantum ML! ✨**

---

*Created for ML beginners learning Quantum Bayesian Classifiers*
*Based on research best practices and production code standards*
