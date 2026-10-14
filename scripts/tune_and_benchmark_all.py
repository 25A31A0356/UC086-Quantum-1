"""
CLI Tool: Complete 3-Model Benchmark Sweep on TCGA-LUAD (Domain 2).
Compares Quantum Kernel SVM (QSVC), Classical SVM (RBF), and Classical XGBoost.
"""

import sys
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.data.tcga_loader import fetch_tcga_luad_cohort
from src.classical.baselines import train_classical_svm, train_classical_xgboost
from src.quantum.feature_maps import build_cancer_feature_map
from src.quantum.quantum_kernel import QuantumKernelEngine, train_qsvc
from src.evaluation.metrics import compute_cancer_metrics, format_benchmark_table

def main():
    print("=" * 80)
    print("  EXHAUSTIVE 3-MODEL BENCHMARK: TCGA-LUAD EGFR MUTATION PREDICTION (DOMAIN 2)")
    print("=" * 80)
    
    df = fetch_tcga_luad_cohort(cache_dir="data")
    feature_cols = [
        'EGFR_mRNA', 'ERBB2_mRNA', 'KRAS_mRNA', 'BRAF_mRNA', 'MET_mRNA',
        'PIK3CA_mRNA', 'STK11_mRNA', 'TP53_mRNA', 'KEAP1_mRNA', 'PTEN_mRNA'
    ]
    X_raw = df[feature_cols].values
    y_raw = df['EGFR_mutant'].values
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y_raw, test_size=0.15, random_state=42, stratify=y_raw
    )
    
    scaler_std = StandardScaler()
    scaler_q = MinMaxScaler(feature_range=(0, np.pi))
    
    X_train_std = scaler_std.fit_transform(X_train_raw)
    X_test_std  = scaler_std.transform(X_test_raw)
    
    X_train_q = scaler_q.fit_transform(X_train_std)
    X_test_q  = scaler_q.transform(X_test_std)
    
    results = []
    
    # 1. Classical RBF SVM
    print("[+] Model 1/3: Classical SVM (RBF Kernel)...")
    svm_model, svm_params, t_svm = train_classical_svm(X_train_std, y_train)
    y_pred_svm = svm_model.predict(X_test_std)
    y_prob_svm = svm_model.predict_proba(X_test_std)[:, 1]
    results.append(compute_cancer_metrics(y_test, y_pred_svm, y_prob_svm, model_name="Classical SVM (RBF)", runtime_sec=t_svm))
    
    # 2. Classical XGBoost
    print("[+] Model 2/3: Classical XGBoost Classifier...")
    xgb_model, xgb_params, t_xgb = train_classical_xgboost(X_train_std, y_train)
    y_pred_xgb = xgb_model.predict(X_test_std)
    y_prob_xgb = xgb_model.predict_proba(X_test_std)[:, 1]
    results.append(compute_cancer_metrics(y_test, y_pred_xgb, y_prob_xgb, model_name="Classical XGBoost", runtime_sec=t_xgb))
    
    # 3. Quantum Kernel SVM (QSVC)
    print("[+] Model 3/3: Quantum Kernel SVM (Qiskit QSVC)...")
    feature_map = build_cancer_feature_map(n_qubits=10, reps=2, entanglement='linear')
    engine = QuantumKernelEngine(feature_map)
    K_train, psi_train, t_k_train = engine.evaluate_train_kernel(X_train_q)
    K_test, t_k_test = engine.evaluate_test_kernel(X_test_q, psi_train)
    qsvc_model, t_qsvc_fit = train_qsvc(K_train, y_train, C=1.0, class_weight='balanced')
    y_pred_qsvc = qsvc_model.predict(K_test)
    y_prob_qsvc = qsvc_model.predict_proba(K_test)[:, 1]
    results.append(compute_cancer_metrics(y_test, y_pred_qsvc, y_prob_qsvc, model_name="Quantum Kernel SVM (Qiskit QSVC)", runtime_sec=t_k_train + t_qsvc_fit))
    
    # 4. Mathematical Kernel-Target Alignment
    print("\n[+] Computing Mathematical Kernel-Target Alignment A(K, yy^T)...")
    from sklearn.metrics.pairwise import rbf_kernel
    y_lbl = np.where(y_train == 1, 1.0, -1.0)
    Y_mat = np.outer(y_lbl, y_lbl)
    
    gamma = 1.0 / (X_train_std.shape[1] * X_train_std.var())
    K_rbf = rbf_kernel(X_train_std, X_train_std, gamma=gamma)
    
    align_rbf = np.sum(K_rbf * Y_mat) / (np.linalg.norm(K_rbf, 'fro') * np.linalg.norm(Y_mat, 'fro'))
    align_quantum = np.sum(K_train * Y_mat) / (np.linalg.norm(K_train, 'fro') * np.linalg.norm(Y_mat, 'fro'))
    align_gain = ((align_quantum - align_rbf) / abs(align_rbf)) * 100
    
    print("=" * 80)
    print("  MATHEMATICAL PROOF: KERNEL-TARGET ALIGNMENT COMPARISON")
    print("=" * 80)
    print(f"Classical RBF Kernel Alignment:    {align_rbf:.4f}")
    print(f"Quantum ZZFeatureMap Alignment:    {align_quantum:.4f}")
    print(f"Empirical Quantum Alignment Gain:  +{align_gain:.1f}%")
    
    print("\n" + "=" * 80)
    print("  FINAL FAIR BENCHMARK EVALUATION (UNTOUCHED TEST SET)")
    print("=" * 80)
    df_res = format_benchmark_table(results)
    print(df_res.to_string(index=False))
    
    os.makedirs("reports", exist_ok=True)
    df_res.to_csv("reports/domain2_benchmark_results.csv", index=False)
    print(f"\n[✓] Benchmark results exported to: reports/domain2_benchmark_results.csv")

if __name__ == "__main__":
    main()
