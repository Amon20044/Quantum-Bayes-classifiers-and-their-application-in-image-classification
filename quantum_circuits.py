"""
Quantum circuit builder module using MindQuantum.
Constructs quantum circuits for different QBC types.
"""

import numpy as np
from mindquantum import *
from mindquantum.core.gates import RY, X
from mindquantum.core.circuit import Circuit
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')


def f_angle(p: float) -> float:
    """
    Encoding function from probability to rotation angle.
    
    From paper Eq. 14: f(P) = 2 * arccos(√P)
    
    This ensures cos²(θ/2) = P for amplitude encoding.
    
    Args:
        p: Probability value in [0, 1]
        
    Returns:
        Rotation angle θ
    """
    p = np.clip(p, 1e-12, 1.0)  # Numerical stability
    return 2 * np.arccos(np.sqrt(p))


class QuantumCircuitBuilder:
    """Base class for building quantum circuits."""
    
    def __init__(self, n_qubits: int):
        """
        Initialize circuit builder.
        
        Args:
            n_qubits: Total number of qubits (label + attributes)
        """
        self.n_qubits = n_qubits
        self.circuit = None
        
    def _apply_controlled_ry_on_state(self, circuit: Circuit, control_qubits: List[int],
                                      target_qubit: int, angle: float, 
                                      control_state: List[int]):
        """
        Apply controlled-RY gate for specific control state.
        
        For control_state with 0s, we flip those controls before and after.
        
        Args:
            circuit: Circuit to modify
            control_qubits: List of control qubit indices
            target_qubit: Target qubit index
            angle: Rotation angle
            control_state: Desired control state (list of 0s and 1s)
        """
        # Flip controls that should be 0
        for i, (ctrl, state) in enumerate(zip(control_qubits, control_state)):
            if state == 0:
                circuit += X.on(ctrl)
        
        # Apply controlled RY
        if len(control_qubits) == 1:
            circuit += RY(angle).on(target_qubit, control_qubits[0])
        elif len(control_qubits) == 2:
            circuit += RY(angle).on(target_qubit, control_qubits)
        else:
            raise ValueError(f"Unsupported number of controls: {len(control_qubits)}")
        
        # Flip back
        for i, (ctrl, state) in enumerate(zip(control_qubits, control_state)):
            if state == 0:
                circuit += X.on(ctrl)


class NaiveQBC(QuantumCircuitBuilder):
    """Naive Quantum Bayes Classifier."""
    
    def __init__(self, n_attributes: int):
        """
        Initialize Naive QBC.
        
        Args:
            n_attributes: Number of attribute qubits
        """
        super().__init__(n_attributes + 1)  # +1 for label qubit
        self.n_attributes = n_attributes
        
    def build_circuit(self, P_y: Dict, cond_probs: Dict) -> Circuit:
        """
        Build Naive QBC circuit.
        
        Qubit layout: q0 = label (y), q1..qn = attributes (x1..xn)
        
        Args:
            P_y: Prior probabilities {class: P(y=class)}
            cond_probs: Conditional probabilities {class: {attr_idx: P(x_i=0|y=class)}}
            
        Returns:
            MindQuantum Circuit
        """
        circuit = Circuit()
        
        # Extract actual class labels (handles any class values, not just 0 and 1)
        class_labels = sorted(list(P_y.keys()))  # e.g., [3, 7] or [0, 1]
        class_0, class_1 = class_labels[0], class_labels[1]
        
        # Encode prior P(y=class_0)
        theta_y = f_angle(P_y[class_0])
        circuit += RY(theta_y).on(0)
        
        # For each attribute
        for attr_idx in range(self.n_attributes):
            qubit_idx = attr_idx + 1  # Offset by 1 (q0 is label)
            
            # Get angles for both classes
            theta_y0 = f_angle(cond_probs[class_0][attr_idx])  # P(x_i=0 | y=class_0)
            theta_y1 = f_angle(cond_probs[class_1][attr_idx])  # P(x_i=0 | y=class_1)
            
            # Controlled RY with y=1 (control = |1⟩)
            circuit += RY(theta_y1).on(qubit_idx, 0)
            
            # Controlled RY with y=0 (control = |0⟩)
            circuit += X.on(0)  # Flip to make |0⟩ → |1⟩
            circuit += RY(theta_y0).on(qubit_idx, 0)
            circuit += X.on(0)  # Flip back
        
        self.circuit = circuit
        return circuit
    
    def get_circuit_summary(self) -> str:
        """Get human-readable circuit summary."""
        if self.circuit is None:
            return "Circuit not built yet"
        
        summary = f"Naive QBC Circuit\n"
        summary += f"  Qubits: {self.n_qubits} (1 label + {self.n_attributes} attributes)\n"
        summary += f"  Gates: {len(self.circuit)} total\n"
        summary += f"  Depth: {self.circuit.depth()}\n"
        return summary


class SPODE_QBC(QuantumCircuitBuilder):
    """SPODE Quantum Bayes Classifier with super-parent."""
    
    def __init__(self, n_attributes: int, super_parent_idx: int = 4):
        """
        Initialize SPODE QBC.
        
        Args:
            n_attributes: Number of attributes
            super_parent_idx: Index of super-parent attribute (default 4 = center of 3x3)
        """
        super().__init__(n_attributes + 1)
        self.n_attributes = n_attributes
        self.super_parent_idx = super_parent_idx
        
    def build_circuit(self, P_y: Dict, super_parent_probs: Dict, cond_probs: Dict) -> Circuit:
        """
        Build SPODE QBC circuit.
        
        Args:
            P_y: Prior probabilities
            super_parent_probs: P(x_super | y)
            cond_probs: P(x_i | y, x_super) for non-super attributes
            
        Returns:
            MindQuantum Circuit
        """
        circuit = Circuit()
        
        # Extract actual class labels
        class_labels = sorted(list(P_y.keys()))
        class_0, class_1 = class_labels[0], class_labels[1]
        
        # Encode prior P(y=class_0)
        theta_y = f_angle(P_y[class_0])
        circuit += RY(theta_y).on(0)
        
        # Encode super-parent (same as Naive for this attribute)
        super_qubit = self.super_parent_idx + 1
        theta_super_y0 = f_angle(super_parent_probs[class_0])
        theta_super_y1 = f_angle(super_parent_probs[class_1])
        
        circuit += RY(theta_super_y1).on(super_qubit, 0)
        circuit += X.on(0)
        circuit += RY(theta_super_y0).on(super_qubit, 0)
        circuit += X.on(0)
        
        # For other attributes: P(x_i | y, x_super) - need 4 controlled-RY gates
        for attr_idx in range(self.n_attributes):
            if attr_idx == self.super_parent_idx:
                continue  # Skip super-parent itself
            
            qubit_idx = attr_idx + 1
            
            # Four configurations: (y, x_super) = (class_0,0), (class_0,1), (class_1,0), (class_1,1)
            for y_val in class_labels:
                for super_val in [0, 1]:
                    p_attr_0 = cond_probs[y_val][attr_idx][super_val]
                    theta = f_angle(p_attr_0)
                    
                    # Apply controlled-RY with controls = [y_qubit, super_qubit]
                    # Control state = [y_val, super_val]
                    # Note: for y_val, we need to map to 0 or 1 position
                    y_state = 0 if y_val == class_0 else 1
                    self._apply_controlled_ry_on_state(
                        circuit, 
                        control_qubits=[0, super_qubit],
                        target_qubit=qubit_idx,
                        angle=theta,
                        control_state=[y_state, super_val]
                    )
        
        self.circuit = circuit
        return circuit
    
    def get_circuit_summary(self) -> str:
        """Get human-readable circuit summary."""
        if self.circuit is None:
            return "Circuit not built yet"
        
        summary = f"SPODE QBC Circuit\n"
        summary += f"  Qubits: {self.n_qubits} (1 label + {self.n_attributes} attributes)\n"
        summary += f"  Super-parent: x{self.super_parent_idx} (qubit {self.super_parent_idx + 1})\n"
        summary += f"  Gates: {len(self.circuit)} total\n"
        summary += f"  Depth: {self.circuit.depth()}\n"
        return summary


class TAN_QBC(QuantumCircuitBuilder):
    """Tree-Augmented Naive Bayes QBC."""
    
    def __init__(self, n_attributes: int):
        """Initialize TAN QBC."""
        super().__init__(n_attributes + 1)
        self.n_attributes = n_attributes
        self.tree_structure = None
        
    def build_circuit(self, P_y: Dict, tree_structure: Dict, cond_probs: Dict) -> Circuit:
        """
        Build TAN QBC circuit.
        
        Args:
            P_y: Prior probabilities
            tree_structure: {child_idx: parent_idx or None}
            cond_probs: Conditional probabilities
            
        Returns:
            MindQuantum Circuit
        """
        circuit = Circuit()
        self.tree_structure = tree_structure
        
        # Encode prior P(y=0)
        # Extract actual class labels
        class_labels = sorted(list(P_y.keys()))
        class_0, class_1 = class_labels[0], class_labels[1]
        
        theta_y = f_angle(P_y[class_0])
        circuit += RY(theta_y).on(0)
        
        # Find root and build layer by layer (BFS)
        root_idx = [i for i, parent in tree_structure.items() if parent is None][0]
        
        # Process root
        root_qubit = root_idx + 1
        theta_root_y0 = f_angle(cond_probs[class_0][root_idx][None])
        theta_root_y1 = f_angle(cond_probs[class_1][root_idx][None])
        
        circuit += RY(theta_root_y1).on(root_qubit, 0)
        circuit += X.on(0)
        circuit += RY(theta_root_y0).on(root_qubit, 0)
        circuit += X.on(0)
        
        # Process non-root nodes
        for attr_idx in range(self.n_attributes):
            parent_idx = tree_structure[attr_idx]
            if parent_idx is None:
                continue  # Skip root (already processed)
            
            qubit_idx = attr_idx + 1
            parent_qubit = parent_idx + 1
            
            # Four configurations: (y, x_parent) = (class_0,0), (class_0,1), (class_1,0), (class_1,1)
            for y_val in class_labels:
                for parent_val in [0, 1]:
                    p_attr_0 = cond_probs[y_val][attr_idx][parent_val]
                    theta = f_angle(p_attr_0)
                    
                    y_state = 0 if y_val == class_0 else 1
                    self._apply_controlled_ry_on_state(
                        circuit,
                        control_qubits=[0, parent_qubit],
                        target_qubit=qubit_idx,
                        angle=theta,
                        control_state=[y_state, parent_val]
                    )
        
        self.circuit = circuit
        return circuit
    
    def get_circuit_summary(self) -> str:
        """Get human-readable circuit summary."""
        if self.circuit is None:
            return "Circuit not built yet"
        
        summary = f"TAN QBC Circuit\n"
        summary += f"  Qubits: {self.n_qubits} (1 label + {self.n_attributes} attributes)\n"
        if self.tree_structure:
            root = [i for i, p in self.tree_structure.items() if p is None][0]
            summary += f"  Root: x{root}\n"
        summary += f"  Gates: {len(self.circuit)} total\n"
        summary += f"  Depth: {self.circuit.depth()}\n"
        return summary


class SymmetricQBC(QuantumCircuitBuilder):
    """Symmetric Quantum Bayes Classifier."""
    
    def __init__(self, n_attributes: int):
        """Initialize Symmetric QBC."""
        super().__init__(n_attributes + 1)
        self.n_attributes = n_attributes
        self.tree_structure = None
        
    def build_circuit(self, P_y: Dict, tree_structure: Dict, cond_probs: Dict) -> Circuit:
        """
        Build Symmetric QBC circuit.
        
        Similar to TAN but with symmetric relationships.
        
        Args:
            P_y: Prior probabilities
            tree_structure: Symmetric structure
            cond_probs: Conditional probabilities
            
        Returns:
            MindQuantum Circuit
        """
        circuit = Circuit()
        self.tree_structure = tree_structure
        
        # Extract actual class labels
        class_labels = sorted(list(P_y.keys()))
        class_0, class_1 = class_labels[0], class_labels[1]
        
        # Encode prior P(y=class_0)
        theta_y = f_angle(P_y[class_0])
        circuit += RY(theta_y).on(0)
        
        # Process each attribute
        for attr_idx in range(self.n_attributes):
            qubit_idx = attr_idx + 1
            parent_idx = tree_structure[attr_idx]
            
            if parent_idx is None:
                # Independent node: P(x_i | y)
                theta_y0 = f_angle(cond_probs[class_0][attr_idx][None])
                theta_y1 = f_angle(cond_probs[class_1][attr_idx][None])
                
                circuit += RY(theta_y1).on(qubit_idx, 0)
                circuit += X.on(0)
                circuit += RY(theta_y0).on(qubit_idx, 0)
                circuit += X.on(0)
            else:
                # Has parent: P(x_i | y, x_parent)
                parent_qubit = parent_idx + 1
                
                for y_val in class_labels:
                    for parent_val in [0, 1]:
                        p_attr_0 = cond_probs[y_val][attr_idx][parent_val]
                        theta = f_angle(p_attr_0)
                        
                        y_state = 0 if y_val == class_0 else 1
                        self._apply_controlled_ry_on_state(
                            circuit,
                            control_qubits=[0, parent_qubit],
                            target_qubit=qubit_idx,
                            angle=theta,
                            control_state=[y_state, parent_val]
                        )
        
        self.circuit = circuit
        return circuit
    
    def get_circuit_summary(self) -> str:
        """Get human-readable circuit summary."""
        if self.circuit is None:
            return "Circuit not built yet"
        
        summary = f"Symmetric QBC Circuit\n"
        summary += f"  Qubits: {self.n_qubits} (1 label + {self.n_attributes} attributes)\n"
        summary += f"  Gates: {len(self.circuit)} total\n"
        summary += f"  Depth: {self.circuit.depth()}\n"
        return summary


def build_qbc_circuit(qbc_type: str, n_attributes: int, statistics, **kwargs) -> Tuple[Circuit, QuantumCircuitBuilder]:
    """
    Factory function to build QBC circuit.
    
    Args:
        qbc_type: One of ['naive', 'spode', 'tan', 'symmetric']
        n_attributes: Number of attributes
        statistics: Fitted statistics object
        **kwargs: Additional arguments (e.g., super_parent_idx for SPODE)
        
    Returns:
        circuit: Built MindQuantum circuit
        builder: Circuit builder object
    """
    qbc_type = qbc_type.lower()
    
    if qbc_type == 'naive':
        builder = NaiveQBC(n_attributes)
        P_y, cond_probs = statistics.get_probabilities()
        circuit = builder.build_circuit(P_y, cond_probs)
        
    elif qbc_type == 'spode':
        super_parent_idx = kwargs.get('super_parent_idx', 4)
        builder = SPODE_QBC(n_attributes, super_parent_idx)
        P_y, super_parent_probs, cond_probs = statistics.get_probabilities()
        circuit = builder.build_circuit(P_y, super_parent_probs, cond_probs)
        
    elif qbc_type == 'tan':
        builder = TAN_QBC(n_attributes)
        P_y, tree_structure, cond_probs = statistics.get_probabilities()
        circuit = builder.build_circuit(P_y, tree_structure, cond_probs)
        
    elif qbc_type == 'symmetric':
        builder = SymmetricQBC(n_attributes)
        P_y, tree_structure, cond_probs = statistics.get_probabilities()
        circuit = builder.build_circuit(P_y, tree_structure, cond_probs)
        
    else:
        raise ValueError(f"Unknown QBC type: {qbc_type}")
    
    return circuit, builder
