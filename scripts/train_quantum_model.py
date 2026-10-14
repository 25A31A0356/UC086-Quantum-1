"""
CLI Tool: Train and Evaluate Quantum Support Vector Classifier (QSVC) on TCGA-LUAD.
"""

import sys
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.tcga_loader import fetch_tcga_luad_cohort
from src.quantum.feature_maps import build_cancer_feature_map
from src.quantum.quantum_kernel import QuantumKernelEngine, train_qsvc
from src.evaluation.metrics import compute_cancer_metrics, format_benchmark_table

def main():
    print("=" * 70)
    print("  QUANTUM SUPPORT VECTOR CLASSIFIER (QSVC) - TCGA-LUAD (DOMAIN 2)")
    print("=" * 70)
    
    df = fetch_tcga_luad_cohort(cache_dir="data")
    feature_cols = [
        'EGFR_mRNA', 'ERBB2_mRNA', 'KRAS_mRNA', 'BRAF_mRNA', 'MET_mRNA',
        'PIK3CA_mRNA', 'STK11_mRNA', 'TP53_mRNA', 'KEAP1_mRNA', 'PTEN_mRNA'
    ]
    X_raw = df[feature_cols].values
    y_raw = df['EGFR_mutant'].values
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.20, random_state=42, stratify=y_raw
    )
    
    scaler_std = StandardScaler()
    scaler_q = MinMaxScaler(feature_range=(0, np.pi))
    
    X_train_std = scaler_std.fit_transform(X_train_raw)
    X_test_std  = scaler_std.transform(X_test_raw)
    
    X_train_q = scaler_q.fit_transform(X_train_std)
    X_test_q  = scaler_q.transform(X_test_std)
    
    print(f"[+] Training samples: {len(X_train_q)} | Test samples: {len(X_test_q)}")
    print(f"[+] Constructing 10-Qubit ZZFeatureMap (reps=2, linear entanglement)...")
    feature_map = build_cancer_feature_map(n_qubits=10, reps=2, entanglement='linear')
    
    engine = QuantumKernelEngine(feature_map)
    print(f"[+] Evaluating Training Gram Matrix (Exact Statevector Fidelity)...")
    K_train, psi_train, t_k_train = engine.evaluate_train_kernel(X_train_q)
    print(f"[✓] K_train computed in {t_k_train:.2f}s (Shape: {K_train.shape})")
    
    print(f"[+] Evaluating Test Gram Matrix...")
    K_test, t_k_test = engine.evaluate_test_kernel(X_test_q, psi_train)
    print(f"[✓] K_test computed in {t_k_test:.2f}s (Shape: {K_test.shape})")
    
    print(f"[+] Training precomputed QSVC with balanced class weighting...")
    clf, t_fit = train_qsvc(K_train, y_train, C=1.0, class_weight='balanced')
    print(f"[✓] QSVC fitted in {t_fit:.3f}s")
    
    y_pred = clf.predict(K_test)
    y_prob = clf.predict_proba(K_test)[:, 1]
    
    metrics = compute_cancer_metrics(y_test, y_pred, y_prob, model_name="Qiskit QSVC (ZZFeatureMap)", runtime_sec=t_k_train + t_fit)
    print("\n" + format_benchmark_table([metrics]).to_string(index=False))

if __name__ == "__main__":
    main()
