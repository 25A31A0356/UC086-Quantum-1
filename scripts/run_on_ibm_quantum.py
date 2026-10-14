"""
CLI Tool: Deploy and Transpile TCGA-LUAD Cancer Feature Map on IBM Quantum Hardware.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.quantum.feature_maps import build_cancer_feature_map
from src.quantum.ibm_quantum_backend import get_quantum_backend, transpile_circuit_for_hardware

def main():
    print("=" * 70)
    print("  IBM QUANTUM HARDWARE DEPLOYMENT - CANCER PREDICTION (DOMAIN 2)")
    print("=" * 70)
    
    backend, is_real_qpu = get_quantum_backend(min_qubits=10)
    print(f"[✓] Target Backend: {backend.name} (Real Hardware: {is_real_qpu})")
    
    feature_map = build_cancer_feature_map(n_qubits=10, reps=2, entanglement='linear')
    sample_angles = np.random.uniform(0, np.pi, 10)
    
    qc = feature_map.assign_parameters(sample_angles)
    qc.measure_all()
    
    print(f"[+] Transpiling 10-qubit cancer circuit with preset pass manager (level=3)...")
    transpiled_qc = transpile_circuit_for_hardware(qc, backend, optimization_level=3)
    
    print(f"[✓] Virtual depth: {feature_map.depth()} -> Physical depth: {transpiled_qc.depth()}")
    print(f"[✓] Physical gate count: {dict(transpiled_qc.count_ops())}")
    
    print(f"[+] Executing 1,024 shots readout...")
    res = backend.run(transpiled_qc, shots=1024).result()
    counts = res.get_counts()
    top_bitstrings = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
    print(f"[✓] Top-5 Measurement Bitstrings: {top_bitstrings}")

if __name__ == "__main__":
    main()
