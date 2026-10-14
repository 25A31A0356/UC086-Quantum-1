"""
Quantum Feature Maps for Cancer Genomics (Qiskit 1.x).
Encodes 10 hallmark oncogenes into a 10-qubit entangled quantum state.
"""

from qiskit.circuit.library import ZZFeatureMap, PauliFeatureMap

def build_cancer_feature_map(n_qubits=10, reps=2, entanglement='linear'):
    """
    Constructs a ZZFeatureMap for mapping cancer gene expression profiles to quantum Hilbert space.
    """
    return ZZFeatureMap(
        feature_dimension=n_qubits,
        reps=reps,
        entanglement=entanglement
    )

def build_pauli_feature_map(n_qubits=10, paulis=['Z', 'ZZ'], reps=2, entanglement='linear'):
    """
    Constructs custom PauliFeatureMap with user-specified Pauli operators.
    """
    return PauliFeatureMap(
        feature_dimension=n_qubits,
        paulis=paulis,
        reps=reps,
        entanglement=entanglement
    )
