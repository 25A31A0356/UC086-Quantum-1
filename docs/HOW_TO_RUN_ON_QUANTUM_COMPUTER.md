# 🧊 How to Run on Real IBM Quantum Hardware (QPU)
### **Domain 2: Quantum Kernel Learning for Cancer Prediction**
**Institution**: Pragati Engineering College | **Qiskit Fall Fest 2026**  
**Framework**: Qiskit 1.x & `qiskit-ibm-runtime` (V2 Primitives)  

---

## 📌 Overview
This guide provides step-by-step instructions for executing the Domain 2 cancer genomics quantum feature map and kernel estimation on **real superconducting quantum processors** provided by IBM Quantum (such as 127-qubit IBM Quantum Eagle or 133-qubit Heron processors).

---

## 🔑 1. Obtain Your Free IBM Quantum API Token
1. Go to the [IBM Quantum Platform](https://quantum.ibm.com/).
2. Create a free account or sign in with your IBM ID.
3. On your dashboard, locate the **API Token** section in the top-right corner.
4. Click **Copy Token** to copy your unique API token to your clipboard.

---

## 🚀 2. Running in Google Colab (`Run_In_The_Quantum_Computer.ipynb`)

### Option A: Interactive Token Input (Recommended)
Open **[Run_In_The_Quantum_Computer.ipynb](https://colab.research.google.com/github/25A31A0356/UC086-Quantum-1/blob/main/notebooks/Run_In_The_Quantum_Computer.ipynb)** in Google Colab.

In **Cell 7 (Physical Quantum Hardware Deployment)**, paste your token when prompted:
```python
import getpass
from qiskit_ibm_runtime import QiskitRuntimeService

# Paste your free IBM Quantum token
IBM_TOKEN = getpass.getpass("Enter your IBM Quantum API Token: ")

# Authenticate with IBM Quantum Cloud
service = QiskitRuntimeService(channel="ibm_quantum", token=IBM_TOKEN)

# Select the least busy operational superconducting quantum processor
backend = service.least_busy(operational=True, simulator=False, min_num_qubits=10)
print(f"[✓] Connected to Live IBM Quantum QPU: {backend.name} ({backend.num_qubits} Qubits)")
```

### Option B: Colab Secrets (Zero Code Exposure)
1. In Google Colab, click the 🔑 **Secrets** icon on the left sidebar.
2. Add a new secret named `IBM_QUANTUM_TOKEN` and paste your token value.
3. Toggle on **Notebook access**.
4. In your notebook cell:
```python
from google.colab import userdata
from qiskit_ibm_runtime import QiskitRuntimeService

token = userdata.get('IBM_QUANTUM_TOKEN')
service = QiskitRuntimeService(channel="ibm_quantum", token=token)
backend = service.least_busy(operational=True, simulator=False, min_num_qubits=10)
print(f"[✓] Connected to QPU: {backend.name}")
```

---

## 💻 3. Running from Local Terminal (CLI)

You can execute the automated quantum hardware deployment script directly via PowerShell:

```powershell
# Set your token as an environment variable (session-scoped)
$env:IBM_QUANTUM_TOKEN = "your_actual_ibm_quantum_api_token_here"

# Execute the deployment script
python scripts/run_on_ibm_quantum.py
```

If no token is provided, the script gracefully falls back to `AerSimulator` mimicking IBM Falcon/Eagle architecture.

---

## ⚛️ 4. Quantum Hardware Pipeline Stages

### Stage 1: Parameterized Feature Map Binding
Patient genomic z-score angles ($x \in [0, \pi]^{10}$) are assigned to the 10-qubit $ZZFeatureMap$:
```python
from src.quantum.feature_maps import build_cancer_feature_map

feature_map = build_cancer_feature_map(n_qubits=10, reps=2, entanglement='linear')
qc = feature_map.assign_parameters(patient_angles)
qc.measure_all()
```

### Stage 2: Hardware Transpilation & Basis Gate Decomposition
Real IBM QPUs support native basis gates such as `['ecr', 'id', 'rz', 'sx', 'x']` or `['cz', 'rz', 'sx', 'x']` on heavy-hex coupling graphs. We transpile with optimization level 3:
```python
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

pm = generate_preset_pass_manager(optimization_level=3, backend=backend)
transpiled_qc = pm.run(qc)

print(f"Virtual Depth:  {feature_map.depth()}")
print(f"Physical Depth: {transpiled_qc.depth()}")
```

### Stage 3: Qiskit 1.x Primitives Execution (`SamplerV2`)
We submit jobs to the IBM Quantum Runtime using the new V2 primitive interface:
```python
from qiskit_ibm_runtime import SamplerV2 as Sampler

sampler = Sampler(mode=backend)
job = sampler.run([transpiled_qc], shots=1024)
print(f"Job ID: {job.job_id()}")
result = job.result()
```

---

## 🛡️ 5. Error Mitigation on NISQ Processors
When running on physical quantum hardware:
1. **Twirled Readout Error Extinction (TREX)**: Mitigates measurement and readout bit-flip noise.
2. **Dynamical Decoupling (DD)**: Applies CPMG or XY4 spin-echo pulse sequences during qubit idle times to suppress dephasing ($T_2$).
3. **Preset Pass Manager Level 3**: Optimizes routing to minimize SWAP gate insertion on the physical chip topology.

---

## 🏁 6. Troubleshooting
* **`QiskitRuntimeError: No backend matching criteria`**: Ensure your token has active compute minutes. Check [quantum.ibm.com](https://quantum.ibm.com) for queue times and available systems.
* **`Queue time is long`**: Use `service.least_busy(operational=True, simulator=False)` to target the QPU with the smallest pending job queue, or test immediately on `ibm_osaka` or `AerSimulator`.
