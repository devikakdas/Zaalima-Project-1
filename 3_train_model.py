"""
FactoryGuard AI - Model Training Pipeline
Production-grade training with imbalance handling, hyperparameter tuning, and evaluation
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (precision_recall_curve, average_precision_score,
                             roc_auc_score, classification_report, confusion_matrix)
from sklearn.linear_model import LogisticRegression
import joblib
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class ProductionModelTrainer:
    """
    Production model training with:
    - Time-series aware train/test split
    - Class imbalance handling
    - Hyperparameter tuning
    - Cost-weighted evaluation
    - Model versioning
    """

    def __init__(self, false_positive_cost=5000, false_negative_cost=2000000):
        self.fp_cost = false_positive_cost
        self.fn_cost = false_negative_cost
        self.models = {}
        self.metrics = {}
        self.optimal_threshold = 0.5

    def prepare_data(self, df, test_size=0.2):
        """
        Stratified time-series split: ensures both classes in test set
        Uses later data for testing while maintaining class balance
        """
        df = df.sort_values('timestamp')

        # Feature columns (exclude metadata and target)
        feature_cols = [col for col in df.columns
                        if col not in ['timestamp', 'robot_id', 'failure_within_24h']]

        # Separate failures and normal operations
        failures = df[df['failure_within_24h'] == 1]
        normals = df[df['failure_within_24h'] == 0]

        print(f"\nTotal dataset: {len(df):,} samples")
        print(f"  Failures: {len(failures):,} ({len(failures) / len(df):.2%})")
        print(f"  Normal: {len(normals):,} ({len(normals) / len(df):.2%})")

        # Time-based split for each class (use last 20% for testing)
        n_test_failures = max(int(len(failures) * test_size), 50)  # At least 50 failures
        n_test_normals = int(len(normals) * test_size)

        # Get latest samples from each class
        test_failures = failures.tail(n_test_failures)
        test_normals = normals.tail(n_test_normals)

        # Get remaining samples for training
        train_failures = failures.head(len(failures) - n_test_failures)
        train_normals = normals.head(len(normals) - n_test_normals)

        # Combine train and test sets
        train_df = pd.concat([train_failures, train_normals]).sort_values('timestamp')
        test_df = pd.concat([test_failures, test_normals]).sort_values('timestamp')

        X_train = train_df[feature_cols]
        y_train = train_df['failure_within_24h']
        X_test = test_df[feature_cols]
        y_test = test_df['failure_within_24h']

        print(f"\n✓ Train set: {len(X_train):,} samples ({y_train.mean():.2%} failure rate)")
        print(f"✓ Test set:  {len(X_test):,} samples ({y_test.mean():.2%} failure rate)")

        # Verify both classes present
        if len(y_test.unique()) < 2:
            raise ValueError(f"Test set has only one class! Classes present: {y_test.unique()}")

        return X_train, X_test, y_train, y_test, feature_cols

    def train_baseline(self, X_train, y_train):
        """Train baseline Logistic Regression"""
        print("\n" + "=" * 60)
        print("TRAINING BASELINE: Logistic Regression")
        print("=" * 60)

        # Calculate class weights for imbalance
        pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

        model = LogisticRegression(
            class_weight={0: 1, 1: pos_weight},
            max_iter=1000,
            random_state=42
        )
        model.fit(X_train, y_train)

        self.models['baseline'] = model
        print("✓ Baseline model trained")

        return model

    def train_lightgbm(self, X_train, y_train, tune_hyperparams=True):
        """Train LightGBM with hyperparameter tuning"""
        print("\n" + "=" * 60)
        print("TRAINING PRODUCTION MODEL: LightGBM")
        print("=" * 60)

        # Calculate scale_pos_weight for imbalance
        pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

        if tune_hyperparams:
            print("Running hyperparameter tuning with TimeSeriesSplit...")

            param_grid = {
                'num_leaves': [31, 50, 70],
                'max_depth': [5, 7, 9],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [100, 200],
                'scale_pos_weight': [pos_weight * 0.5, pos_weight, pos_weight * 1.5]
            }

            base_model = lgb.LGBMClassifier(
                objective='binary',
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )

            # Time series cross-validation
            tscv = TimeSeriesSplit(n_splits=3)

            grid_search = GridSearchCV(
                base_model,
                param_grid,
                cv=tscv,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )

            grid_search.fit(X_train, y_train)

            model = grid_search.best_estimator_
            print(f"\n✓ Best parameters: {grid_search.best_params_}")
            print(f"✓ Best CV ROC-AUC: {grid_search.best_score_:.4f}")

        else:
            # Use good default parameters
            model = lgb.LGBMClassifier(
                num_leaves=50,
                max_depth=7,
                learning_rate=0.05,
                n_estimators=200,
                scale_pos_weight=pos_weight,
                objective='binary',
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            model.fit(X_train, y_train)
            print("✓ Model trained with default parameters")

        self.models['lightgbm'] = model

        return model

    def evaluate_model(self, model, X_test, y_test, model_name):
        """Comprehensive model evaluation"""
        print(f"\n{'=' * 60}")
        print(f"EVALUATION: {model_name}")
        print("=" * 60)

        # Predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= self.optimal_threshold).astype(int)

        # Metrics
        pr_auc = average_precision_score(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        print(f"\nMetrics (threshold={self.optimal_threshold}):")
        print(f"  PR-AUC (Primary):  {pr_auc:.4f}")
        print(f"  ROC-AUC:           {roc_auc:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        print(f"\nConfusion Matrix:")
        print(f"  True Negatives:  {tn:,}")
        print(f"  False Positives: {fp:,}")
        print(f"  False Negatives: {fn:,}")
        print(f"  True Positives:  {tp:,}")

        # Cost-weighted evaluation
        total_cost = (fp * self.fp_cost) + (fn * self.fn_cost)
        print(f"\nBusiness Impact:")
        print(f"  FP Cost: ${fp * self.fp_cost:,.0f} ({fp} unnecessary maintenances)")
        print(f"  FN Cost: ${fn * self.fn_cost:,.0f} ({fn} missed failures)")
        print(f"  Total Cost: ${total_cost:,.0f}")

        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['Normal', 'Failure']))

        # Store metrics
        self.metrics[model_name] = {
            'pr_auc': float(pr_auc),
            'roc_auc': float(roc_auc),
            'confusion_matrix': cm.tolist(),
            'total_cost': int(total_cost),
            'threshold': float(self.optimal_threshold),
            'timestamp': datetime.now().isoformat()
        }

        return pr_auc, roc_auc, total_cost

    def optimize_threshold(self, model, X_test, y_test):
        """Find optimal threshold to minimize business cost"""
        print("\n" + "=" * 60)
        print("OPTIMIZING DECISION THRESHOLD")
        print("=" * 60)

        y_pred_proba = model.predict_proba(X_test)[:, 1]

        precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

        min_cost = float('inf')
        optimal_threshold = 0.5

        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()

            cost = (fp * self.fp_cost) + (fn * self.fn_cost)

            if cost < min_cost:
                min_cost = cost
                optimal_threshold = threshold

        self.optimal_threshold = optimal_threshold

        print(f"✓ Optimal threshold: {optimal_threshold:.4f}")
        print(f"✓ Minimum cost: ${min_cost:,.0f}")

        return optimal_threshold


# Load engineered features
print("Loading engineered features...")
df = pd.read_csv('sensor_data_features.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"✓ Loaded {len(df):,} records with {df.shape[1]} features")

# Initialize trainer
trainer = ProductionModelTrainer(
    false_positive_cost=5000,  # Unnecessary maintenance
    false_negative_cost=2000000  # Catastrophic failure
)

# Prepare data (stratified split ensures both classes in test set)
X_train, X_test, y_train, y_test, feature_cols = trainer.prepare_data(df, test_size=0.2)

# Train baseline
baseline_model = trainer.train_baseline(X_train, y_train)

# Train LightGBM (set tune_hyperparams=False for faster execution)
lgb_model = trainer.train_lightgbm(X_train, y_train, tune_hyperparams=False)

# Optimize threshold
trainer.optimize_threshold(lgb_model, X_test, y_test)

# Evaluate both models
trainer.evaluate_model(baseline_model, X_test, y_test, 'Logistic Regression (Baseline)')
trainer.evaluate_model(lgb_model, X_test, y_test, 'LightGBM (Production)')

# Save production model
print("\n" + "=" * 60)
print("SAVING PRODUCTION ARTIFACTS")
print("=" * 60)

joblib.dump(lgb_model, 'models/lightgbm_model.joblib', compress=3)
print("✓ Model saved: models/lightgbm_model.joblib")

# Save metadata
metadata = {
    'model_type': 'LightGBM',
    'feature_count': len(feature_cols),
    'feature_names': feature_cols,
    'optimal_threshold': float(trainer.optimal_threshold),
    'metrics': trainer.metrics['LightGBM (Production)'],
    'training_date': datetime.now().isoformat(),
    'class_weights': f"scale_pos_weight={lgb_model.scale_pos_weight}"
}

with open('models/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("✓ Metadata saved: models/model_metadata.json")

print("\n" + "=" * 60)
print("MODEL TRAINING COMPLETE")
print("=" * 60)
print(f"✓ Production model ready for deployment")
print(f"✓ Threshold optimized for business cost")
print(f"✓ All artifacts saved to models/ directory")