"""
Quantum Kernel Computation and QSVC for TCGA-LUAD Cancer Prediction (Qiskit 1.x).
Vectorized statevector fidelity computation and precomputed kernel SVM.
"""

import time
import numpy as np
from qiskit.quantum_info import Statevector
from sklearn.svm import SVC

class QuantumKernelEngine:
    """
    Computes exact Fidelity Quantum Kernel matrices using Qiskit Statevectors:
    K(i, j) = |<Phi(x_i)|Phi(x_j)>|^2
    """
    def __init__(self, feature_map):
        self.feature_map = feature_map
        self.num_qubits = feature_map.num_qubits

    def compute_statevectors(self, X):
        """
        Evaluates exact Statevectors for an array of samples X (N, n_qubits).
        """
        n_samples = len(X)
        dim = 2 ** self.num_qubits
        psi_matrix = np.zeros((n_samples, dim), dtype=complex)
        
        for i in range(n_samples):
            bound_qc = self.feature_map.assign_parameters(X[i])
            sv = Statevector.from_instruction(bound_qc)
            psi_matrix[i, :] = sv.data
            
        return psi_matrix

    def evaluate_train_kernel(self, X_train):
        """
        Computes symmetric training Gram matrix K_train (N, N).
        """
        t0 = time.time()
        psi_train = self.compute_statevectors(X_train)
        inner = np.dot(psi_train, psi_train.conj().T)
        K_train = np.abs(inner) ** 2
        np.fill_diagonal(K_train, 1.0)
        elapsed = time.time() - t0
        return K_train, psi_train, elapsed

    def evaluate_test_kernel(self, X_test, psi_train):
        """
        Computes rectangular test Gram matrix K_test (M, N).
        """
        t0 = time.time()
        psi_test = self.compute_statevectors(X_test)
        inner = np.dot(psi_test, psi_train.conj().T)
        K_test = np.abs(inner) ** 2
        elapsed = time.time() - t0
        return K_test, elapsed

def train_qsvc(K_train, y_train, C=1.0, class_weight='balanced', seed=42):
    """
    Trains Quantum Support Vector Classifier (QSVC) on precomputed Gram matrix.
    """
    t0 = time.time()
    clf = SVC(
        C=C,
        kernel='precomputed',
        class_weight=class_weight,
        probability=True,
        random_state=seed
    )
    clf.fit(K_train, y_train)
    elapsed = time.time() - t0
    return clf, elapsed
