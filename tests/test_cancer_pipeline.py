"""
Unit Tests for Domain 2: Quantum Kernel Learning for Cancer Prediction.
"""

import unittest
import numpy as np
from src.data.tcga_loader import fetch_tcga_luad_cohort
from src.classical.baselines import train_classical_svm, train_classical_xgboost
from src.quantum.feature_maps import build_cancer_feature_map
from src.quantum.quantum_kernel import QuantumKernelEngine, train_qsvc
from src.evaluation.metrics import compute_cancer_metrics

class TestDomain2Pipeline(unittest.TestCase):
    def setUp(self):
        self.df = fetch_tcga_luad_cohort(cache_dir="data")
        self.feature_cols = [
            'EGFR_mRNA', 'ERBB2_mRNA', 'KRAS_mRNA', 'BRAF_mRNA', 'MET_mRNA',
            'PIK3CA_mRNA', 'STK11_mRNA', 'TP53_mRNA', 'KEAP1_mRNA', 'PTEN_mRNA'
        ]

    def test_tcga_dataset_shape_and_labels(self):
        self.assertEqual(len(self.df), 566)
        self.assertIn('EGFR_mutant', self.df.columns)
        self.assertEqual(self.df['EGFR_mutant'].nunique(), 2)
        n_mutants = (self.df['EGFR_mutant'] == 1).sum()
        self.assertGreater(n_mutants, 50) # TCGA verified ~70 mutants

    def test_quantum_feature_map(self):
        fmap = build_cancer_feature_map(n_qubits=10, reps=2, entanglement='linear')
        self.assertEqual(fmap.num_qubits, 10)
        self.assertEqual(fmap.num_parameters, 10)

    def test_quantum_kernel_computation(self):
        fmap = build_cancer_feature_map(n_qubits=4, reps=1, entanglement='linear')
        engine = QuantumKernelEngine(fmap)
        X_sample = np.random.uniform(0, np.pi, (5, 4))
        K_train, psi_train, _ = engine.evaluate_train_kernel(X_sample)
        
        # Check shape
        self.assertEqual(K_train.shape, (5, 5))
        # Check diagonal
        np.testing.assert_allclose(np.diag(K_train), 1.0, atol=1e-6)
        # Check symmetry
        np.testing.assert_allclose(K_train, K_train.T, atol=1e-6)

    def test_cancer_metrics(self):
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 0, 0, 0])
        m = compute_cancer_metrics(y_true, y_pred, model_name="TestModel")
        self.assertIn('Accuracy', m)
        self.assertIn('ROC-AUC', m)
        self.assertGreaterEqual(m['Accuracy'], 0.0)

if __name__ == '__main__':
    unittest.main()
