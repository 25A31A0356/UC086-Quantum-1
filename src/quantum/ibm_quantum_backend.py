"""
IBM Quantum QPU Deployment & Runtime Primitives Module (Qiskit 1.x).
Supports transpilation with preset pass managers (optimization_level=3) and SamplerV2 readout.
"""

import os
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator

def get_quantum_backend(token=None, min_qubits=10):
    """
    Connects to IBM Quantum QPU if API token provided, else falls back to AerSimulator.
    """
    if token is None:
        token = os.environ.get("IBM_QUANTUM_TOKEN", "")
        
    if token:
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
            service = QiskitRuntimeService(channel="ibm_quantum", token=token)
            backend = service.least_busy(operational=True, simulator=False, min_num_qubits=min_qubits)
            return backend, True
        except Exception as e:
            print(f"[!] IBM Quantum Cloud notice: {e}")
            
    return AerSimulator(), False

def transpile_circuit_for_hardware(qc, backend, optimization_level=3):
    """
    Transpiles circuit into target backend basis gate set with preset pass manager.
    """
    pm = generate_preset_pass_manager(optimization_level=optimization_level, backend=backend)
    return pm.run(qc)
