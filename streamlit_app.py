"""
Streamlit App for Quantum Bayesian Classifier.
Interactive web interface for QBC experiments on MNIST/Fashion-MNIST.
"""

import streamlit as st
import numpy as np
import sys
import os

# Add project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessing import DataLoader, ImagePreprocessor
from bayesian_stats import NaiveBayesStats, SPODEStats, TANStats, SymmetricStats
from quantum_circuits import build_qbc_circuit
from inference import QBCInference, PerformanceEvaluator
from visualization import QBCVisualizer
from hybrid_classifier import HybridQBCClassifier, EnsembleQBCClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Quantum Bayesian Classifier",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E86DE;
        text-align: center;
        padding: 1rem;
        background: #FF5151;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #FF5151;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background:#FF5151;
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background: #FF5151;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        border: none;
        font-size: 1.1rem;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<h1 class="main-header">🔬 Quantum Bayesian Classifier</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Based on MindQuantum | Reproducing Research Paper Results</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("---")

# Dataset selection
dataset_name = st.sidebar.selectbox(
    "📊 Select Dataset",
    ["MNIST", "Fashion-MNIST"],
    help="Choose between handwritten digits (MNIST) or clothing items (Fashion-MNIST)"
)

# Convert to OpenML name
dataset_openml = "mnist_784" if dataset_name == "MNIST" else "Fashion-MNIST"

# Class selection for binary classification
st.sidebar.subheader("🎯 Binary Classification")
col1, col2 = st.sidebar.columns(2)
with col1:
    class0 = st.number_input("Class 0", min_value=0, max_value=9, value=0)
with col2:
    class1 = st.number_input("Class 1", min_value=0, max_value=9, value=1)

if class0 == class1:
    st.sidebar.error("⚠️ Please select different classes")

# QBC type selection
st.sidebar.subheader("🧠 QBC Architecture")
qbc_types = st.sidebar.multiselect(
    "Select QBC Types to Compare",
    ["Naive", "SPODE", "TAN", "Symmetric"],
    default=["Naive", "SPODE"],
    help="Choose one or more Quantum Bayesian Classifier architectures"
)

# Hybrid approach option
use_hybrid = st.sidebar.checkbox(
    "🔬 Use Hybrid Classical-Quantum Approach",
    value=False,
    help="Combine GaussianNB with quantum circuits for enhanced performance"
)

if use_hybrid:
    st.sidebar.markdown("**Hybrid Configuration:**")
    classical_weight = st.sidebar.slider("Classical Weight", 0.0, 1.0, 0.3, 0.05)
    quantum_weight = 1.0 - classical_weight
    st.sidebar.info(f"Quantum Weight: {quantum_weight:.2f}")
    
    feature_selection = st.sidebar.selectbox(
        "Feature Selection Method",
        ["gaussian_nb", "mutual_info", "f_score"],
        help="Method for selecting most important features"
    )

# Preprocessing parameters
st.sidebar.subheader("🔧 Preprocessing")
block_size = st.sidebar.slider("Sampling Block Size", 5, 9, 7, help="Size of local sampling blocks")
grid_size = st.sidebar.slider("Grid Size", 2, 4, 3, help="Number of blocks per dimension (3x3 = 9 features)")
n_train_samples = st.sidebar.slider("Training Samples per Class", 100, 1000, 500, step=100)
n_test_samples = st.sidebar.slider("Test Samples per Class", 50, 500, 200, step=50)

st.sidebar.markdown("---")
run_experiment = st.sidebar.button("🚀 Run Experiment", use_container_width=True)

# Initialize session state
if 'experiment_run' not in st.session_state:
    st.session_state.experiment_run = False
if 'results' not in st.session_state:
    st.session_state.results = {}

# Main content
if run_experiment:
    if class0 == class1:
        st.error("⚠️ Please select different classes in the sidebar!")
    elif len(qbc_types) == 0:
        st.error("⚠️ Please select at least one QBC type!")
    else:
        st.session_state.experiment_run = True
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Step 1: Load Data
            status_text.text("📥 Loading dataset...")
            progress_bar.progress(10)
            
            loader = DataLoader(dataset_openml)
            X_all, y_all = loader.load_data()
            
            # Filter for binary classification and limit samples
            mask = (y_all == class0) | (y_all == class1)
            X_filtered = X_all[mask]
            y_filtered = y_all[mask]
            
            # Balance classes and limit samples
            mask0 = y_filtered == class0
            mask1 = y_filtered == class1
            
            indices0 = np.where(mask0)[0]
            indices1 = np.where(mask1)[0]
            
            # Training set
            train_indices0 = indices0[:n_train_samples]
            train_indices1 = indices1[:n_train_samples]
            train_indices = np.concatenate([train_indices0, train_indices1])
            np.random.shuffle(train_indices)
            
            # Test set
            test_indices0 = indices0[n_train_samples:n_train_samples+n_test_samples]
            test_indices1 = indices1[n_train_samples:n_train_samples+n_test_samples]
            test_indices = np.concatenate([test_indices0, test_indices1])
            np.random.shuffle(test_indices)
            
            X_train = X_filtered[train_indices]
            y_train = y_filtered[train_indices]
            X_test = X_filtered[test_indices]
            y_test = y_filtered[test_indices]
            
            st.success(f"✅ Loaded {len(X_train)} training and {len(X_test)} test samples")
            
            # Step 2: Preprocess
            status_text.text("⚙️ Preprocessing images...")
            progress_bar.progress(20)
            
            preprocessor = ImagePreprocessor(block_size, grid_size)
            X_train_binary, y_train_binary = preprocessor.fit_transform(X_train, y_train, class0, class1)
            X_test_binary = preprocessor.transform(X_test)
            
            n_attributes = X_train_binary.shape[1]
            st.success(f"✅ Extracted {n_attributes} binary features per image")
            
            # Initialize visualizer
            viz = QBCVisualizer()
            
            # Store results
            results = {}
            
            # Step 3: Train each QBC type
            progress_step = 70 / len(qbc_types)
            
            for idx, qbc_type in enumerate(qbc_types):
                status_text.text(f"🧠 Building {qbc_type} QBC...")
                progress_bar.progress(int(30 + idx * progress_step))
                
                # Compute statistics
                if qbc_type == "Naive":
                    stats = NaiveBayesStats()
                    stats.fit(X_train_binary, y_train, [class0, class1])
                    
                elif qbc_type == "SPODE":
                    super_parent_idx = n_attributes // 2  # Center attribute
                    stats = SPODEStats(super_parent_idx=super_parent_idx)
                    stats.fit(X_train_binary, y_train, [class0, class1])
                    
                elif qbc_type == "TAN":
                    stats = TANStats()
                    stats.fit(X_train_binary, y_train, [class0, class1])
                    
                elif qbc_type == "Symmetric":
                    # Define symmetric pairs for 3x3 grid
                    symmetric_pairs = [(0, 6), (1, 7), (2, 8), (0, 2), (6, 8)]
                    stats = SymmetricStats(symmetric_pairs=symmetric_pairs)
                    stats.fit(X_train_binary, y_train, [class0, class1])
                
                # Build quantum circuit
                circuit, builder = build_qbc_circuit(
                    qbc_type.lower(), n_attributes, stats,
                    super_parent_idx=n_attributes // 2
                )
                
                st.success(f"✅ Built {qbc_type} circuit: {len(circuit)} gates, depth {circuit.depth()}")
                
                # Inference
                status_text.text(f"🔮 Running {qbc_type} inference...")
                
                inference = QBCInference(circuit, n_attributes)
                y_pred, probabilities = inference.predict(X_test_binary, verbose=False)
                
                # Evaluate
                evaluator = PerformanceEvaluator()
                metrics = evaluator.compute_metrics(y_test, y_pred, class0, class1)
                
                # Store results
                results[qbc_type] = {
                    'metrics': metrics,
                    'predictions': y_pred,
                    'probabilities': probabilities,
                    'circuit': circuit,
                    'builder': builder,
                    'stats': stats
                }
                
                st.success(f"✅ {qbc_type} Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
            
            # Save to session state
            st.session_state.results = results
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.X_test_binary = X_test_binary
            st.session_state.class0 = class0
            st.session_state.class1 = class1
            st.session_state.preprocessor = preprocessor
            st.session_state.n_attributes = n_attributes
            
            progress_bar.progress(100)
            status_text.text("✅ Experiment completed!")
            
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)

# Display Results
if st.session_state.experiment_run and st.session_state.results:
    st.markdown("---")
    st.header("📊 Results & Visualizations")
    
    results = st.session_state.results
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test
    X_test_binary = st.session_state.X_test_binary
    class0 = st.session_state.class0
    class1 = st.session_state.class1
    preprocessor = st.session_state.preprocessor
    n_attributes = st.session_state.n_attributes
    
    viz = QBCVisualizer()
    
    # Tabs for organized display
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Performance Comparison", 
        "🔍 Detailed Results",
        "🖼️ Prediction Examples",
        "🌲 Network Structures",
        "📊 Data Visualization"
    ])
    
    with tab1:
        st.subheader("🏆 Performance Comparison")
        
        # Metrics comparison
        metrics_dict = {name: res['metrics'] for name, res in results.items()}
        
        col1, col2, col3, col4 = st.columns(4)
        
        best_acc = max(m['accuracy'] for m in metrics_dict.values())
        best_clf = [name for name, m in metrics_dict.items() if m['accuracy'] == best_acc][0]
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("🏆 Best Classifier", best_clf)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("🎯 Best Accuracy", f"{best_acc:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            avg_acc = np.mean([m['accuracy'] for m in metrics_dict.values()])
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("📊 Average Accuracy", f"{avg_acc:.4f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            total_tests = len(y_test)
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("🧪 Test Samples", total_tests)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Bar chart comparison
        fig_compare = viz.plot_metrics_comparison(
            metrics_dict,
            title=f"QBC Performance: {dataset_name} (Class {class0} vs {class1})"
        )
        st.pyplot(fig_compare)
        
        # Interactive plotly chart
        st.subheader("📊 Interactive Comparison")
        fig_interactive = viz.plot_interactive_comparison(metrics_dict)
        st.plotly_chart(fig_interactive, use_container_width=True)
    
    with tab2:
        st.subheader("🔍 Detailed Results by Classifier")
        
        for name, res in results.items():
            with st.expander(f"📋 {name} QBC Details", expanded=False):
                metrics = res['metrics']
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.markdown("**📊 Metrics:**")
                    st.write(f"- **Accuracy**: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
                    st.write(f"- **Precision**: {metrics['precision']:.4f}")
                    st.write(f"- **Recall**: {metrics['recall']:.4f}")
                    st.write(f"- **F1 Score**: {metrics['f1']:.4f}")
                    st.write(f"- **True Positives**: {metrics['TP']}")
                    st.write(f"- **True Negatives**: {metrics['TN']}")
                    st.write(f"- **False Positives**: {metrics['FP']}")
                    st.write(f"- **False Negatives**: {metrics['FN']}")
                    
                    st.markdown("**🔬 Circuit Info:**")
                    st.write(f"- **Total Qubits**: {res['builder'].n_qubits}")
                    st.write(f"- **Total Gates**: {len(res['circuit'])}")
                    st.write(f"- **Circuit Depth**: {res['circuit'].depth()}")
                
                with col2:
                    # Confusion matrix
                    fig_cm = viz.plot_confusion_matrix(
                        metrics['confusion_matrix'], 
                        class0, class1,
                        title=f"{name} Confusion Matrix"
                    )
                    st.pyplot(fig_cm)
    
    with tab3:
        st.subheader("🖼️ Prediction Examples")
        
        selected_clf = st.selectbox("Select Classifier for Predictions", list(results.keys()))
        
        if selected_clf:
            res = results[selected_clf]
            fig_pred = viz.plot_prediction_examples(
                X_test, y_test, res['predictions'], res['probabilities'],
                n_samples=10,
                title=f"{selected_clf} Prediction Examples"
            )
            st.pyplot(fig_pred)
    
    with tab4:
        st.subheader("🌲 Bayesian Network Structures")
        
        cols = st.columns(len(results))
        
        for idx, (name, res) in enumerate(results.items()):
            with cols[idx]:
                st.markdown(f"**{name} Structure**")
                
                # Get tree structure if applicable
                tree_structure = None
                if hasattr(res['stats'], 'tree_structure'):
                    tree_structure = res['stats'].tree_structure
                
                fig_network = viz.plot_bayesian_network(
                    name.lower(), 
                    tree_structure,
                    n_attributes
                )
                st.pyplot(fig_network)
    
    with tab5:
        st.subheader("📊 Data Visualization")
        
        # Sample images
        fig_samples = viz.plot_sample_images(
            X_test, y_test, n_samples=10,
            title=f"Sample Test Images: {dataset_name}"
        )
        st.pyplot(fig_samples)
        
        st.markdown("---")
        
        # Sampling grid
        st.subheader("🔍 Local Feature Sampling")
        sample_idx = st.slider("Select Image Index", 0, len(X_test)-1, 0)
        
        fig_sampling = viz.plot_sampling_grid(
            X_test[sample_idx],
            preprocessor.sampler.positions,
            preprocessor.sampler.block_size,
            title=f"Sampling on Image {sample_idx} (Label: {y_test[sample_idx]})"
        )
        st.pyplot(fig_sampling)
        
        st.markdown("---")
        
        # Binary features
        st.subheader("🎲 Binary Feature Vectors")
        fig_binary = viz.plot_binary_features(
            X_test_binary, y_test, n_samples=20,
            title="Binarized Feature Vectors"
        )
        st.pyplot(fig_binary)
        
        st.markdown("---")
        
        # Gaussian distributions
        st.subheader("📈 Gaussian Binarization")
        attr_idx = st.slider("Select Attribute", 0, n_attributes-1, 0)
        
        fig_gaussian = viz.plot_gaussian_distributions(
            preprocessor.binarizer.mu0,
            preprocessor.binarizer.sig0,
            preprocessor.binarizer.mu1,
            preprocessor.binarizer.sig1,
            attr_idx=attr_idx,
            title="Gaussian Distributions for Binarization"
        )
        st.pyplot(fig_gaussian)

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 About")
st.sidebar.info("""
**Quantum Bayesian Classifier**

Based on the paper:
*"Quantum Bayes classifiers and their application in image classification"*
by Ming-Ming Wang & Xiao-Ying Zhang

**Features:**
- 🔬 4 QBC architectures (Naive, SPODE, TAN, Symmetric)
- 🎯 Binary image classification
- 📊 Comprehensive metrics & visualizations
- 🌐 Built with MindQuantum

**Implementation:**
- Local feature sampling (3x3 grid)
- Gaussian MLE binarization
- Amplitude encoding with f(P) = 2·arccos(√P)
- Controlled-RY quantum gates
""")

st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ using MindQuantum & Streamlit")
