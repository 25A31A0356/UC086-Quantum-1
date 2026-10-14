"""
Classical Machine Learning Baselines for TCGA-LUAD Cancer Prediction:
1. Classical Support Vector Machine (RBF Kernel)
2. Extreme Gradient Boosting (XGBoost Classifier)
"""

import time
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import xgboost as xgb

def train_classical_svm(X_train, y_train, seed=42):
    """
    Trains non-linear RBF SVM with 5-fold CV hyperparameter tuning and balanced class weights.
    """
    param_grid = {
        'C': [0.1, 1.0, 5.0, 10.0, 50.0],
        'gamma': ['scale', 'auto', 0.05, 0.1, 0.2]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    grid = GridSearchCV(
        SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=seed),
        param_grid,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1
    )
    t0 = time.time()
    grid.fit(X_train, y_train)
    elapsed = time.time() - t0
    return grid.best_estimator_, grid.best_params_, elapsed

def train_classical_xgboost(X_train, y_train, seed=42):
    """
    Trains XGBoost Classifier with scale_pos_weight for imbalanced cancer mutation classification.
    """
    scale_pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    param_grid = {
        'max_depth': [2, 3, 4],
        'learning_rate': [0.03, 0.08, 0.15],
        'n_estimators': [50, 100, 150],
        'subsample': [0.8, 1.0]
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    grid = GridSearchCV(
        xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric='logloss',
            random_state=seed
        ),
        param_grid,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1
    )
    t0 = time.time()
    grid.fit(X_train, y_train)
    elapsed = time.time() - t0
    return grid.best_estimator_, grid.best_params_, elapsed
