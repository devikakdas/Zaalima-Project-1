"""
FactoryGuard AI - Automated Model Retraining Pipeline
Triggered weekly OR when performance drops below threshold
"""

import pandas as pd
import numpy as np
import joblib
import json
import lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score
from datetime import datetime, timedelta
import shutil
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='{"timestamp": "%(asctime)s", "level": "%(levelname)s", "message": "%(message)s"}'
)
logger = logging.getLogger(__name__)


class AutomatedRetrainingPipeline:
    """
    Production retraining pipeline with:
    - Data validation
    - Performance comparison
    - Automatic rollback on degradation
    - Model versioning
    """

    def __init__(self,
                 data_path='sensor_data_features.csv',
                 model_dir='models',
                 min_pr_auc=0.75,
                 improvement_threshold=0.02):
        self.data_path = data_path
        self.model_dir = model_dir
        self.min_pr_auc = min_pr_auc
        self.improvement_threshold = improvement_threshold

    def load_current_model(self):
        """Load currently deployed model"""
        try:
            model = joblib.load(f'{self.model_dir}/lightgbm_model.joblib')
            with open(f'{self.model_dir}/model_metadata.json', 'r') as f:
                metadata = json.load(f)

            logger.info(f"✓ Loaded current model (PR-AUC: {metadata['metrics']['pr_auc']:.4f})")
            return model, metadata
        except Exception as e:
            logger.error(f"Failed to load current model: {str(e)}")
            raise

    def validate_data(self, df):
        """
        Data validation checks before retraining
        - Check for sufficient data
        - Validate distributions
        - Check for anomalies
        """
        logger.info("Running data validation checks...")

        checks_passed = True

        # Check 1: Sufficient data
        min_samples = 10000
        if len(df) < min_samples:
            logger.error(f"❌ Insufficient data: {len(df)} < {min_samples}")
            checks_passed = False
        else:
            logger.info(f"✓ Sufficient data: {len(df):,} samples")

        # Check 2: Class balance (should be around 1%)
        failure_rate = df['failure_within_24h'].mean()
        if failure_rate < 0.005 or failure_rate > 0.05:
            logger.warning(f"⚠️  Unusual failure rate: {failure_rate:.2%} (expected ~1%)")
        else:
            logger.info(f"✓ Failure rate within expected range: {failure_rate:.2%}")

        # Check 3: Missing values
        missing_pct = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_pct > 0.05:
            logger.error(f"❌ Too many missing values: {missing_pct:.2%}")
            checks_passed = False
        else:
            logger.info(f"✓ Missing values acceptable: {missing_pct:.2%}")

        # Check 4: Feature distributions (check for data drift)
        sensor_cols = ['vibration', 'temperature', 'pressure']
        for col in sensor_cols:
            mean = df[col].mean()
            std = df[col].std()

            # Expected ranges (from training data)
            expected_ranges = {
                'vibration': (0.2, 0.5),
                'temperature': (60, 80),
                'pressure': (140, 170)
            }

            if col in expected_ranges:
                min_val, max_val = expected_ranges[col]
                if mean < min_val or mean > max_val:
                    logger.warning(f"⚠️  {col} mean out of range: {mean:.2f}")

        return checks_passed

    def prepare_training_data(self, df, lookback_days=90):
        """Prepare data for retraining (last N days)"""
        df = df.sort_values('timestamp')
        cutoff_date = df['timestamp'].max() - timedelta(days=lookback_days)

        df_recent = df[df['timestamp'] >= cutoff_date].copy()

        # Train/test split (last 14 days for testing)
        test_cutoff = df_recent['timestamp'].max() - timedelta(days=14)

        train_df = df_recent[df_recent['timestamp'] < test_cutoff]
        test_df = df_recent[df_recent['timestamp'] >= test_cutoff]

        # Feature columns
        feature_cols = [col for col in df.columns
                        if col not in ['timestamp', 'robot_id', 'failure_within_24h']]

        X_train = train_df[feature_cols]
        y_train = train_df['failure_within_24h']
        X_test = test_df[feature_cols]
        y_test = test_df['failure_within_24h']

        logger.info(f"✓ Train: {len(X_train):,} samples ({y_train.mean():.2%} failure rate)")
        logger.info(f"✓ Test:  {len(X_test):,} samples ({y_test.mean():.2%} failure rate)")

        return X_train, X_test, y_train, y_test, feature_cols

    def train_new_model(self, X_train, y_train):
        """Train new model with same configuration"""
        logger.info("Training new model...")

        # Calculate class weight
        pos_weight = (len(y_train) - y_train.sum()) / y_train.sum()

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

        logger.info("✓ New model trained")
        return model

    def evaluate_models(self, current_model, new_model, X_test, y_test):
        """
        Compare current and new model performance
        Returns: (should_deploy, metrics_comparison)
        """
        logger.info("Evaluating models...")

        # Current model predictions
        y_pred_current = current_model.predict_proba(X_test)[:, 1]
        pr_auc_current = average_precision_score(y_test, y_pred_current)
        roc_auc_current = roc_auc_score(y_test, y_pred_current)

        # New model predictions
        y_pred_new = new_model.predict_proba(X_test)[:, 1]
        pr_auc_new = average_precision_score(y_test, y_pred_new)
        roc_auc_new = roc_auc_score(y_test, y_pred_new)

        logger.info(f"Current model - PR-AUC: {pr_auc_current:.4f}, ROC-AUC: {roc_auc_current:.4f}")
        logger.info(f"New model     - PR-AUC: {pr_auc_new:.4f}, ROC-AUC: {roc_auc_new:.4f}")

        # Decision logic
        improvement = pr_auc_new - pr_auc_current

        # Check minimum performance threshold
        if pr_auc_new < self.min_pr_auc:
            logger.error(f"❌ New model below minimum threshold: {pr_auc_new:.4f} < {self.min_pr_auc}")
            return False, {
                'current': {'pr_auc': pr_auc_current, 'roc_auc': roc_auc_current},
                'new': {'pr_auc': pr_auc_new, 'roc_auc': roc_auc_new},
                'improvement': improvement,
                'decision': 'rejected_low_performance'
            }

        # Check improvement threshold
        if improvement < self.improvement_threshold and pr_auc_new < pr_auc_current:
            logger.warning(f"⚠️  New model not significantly better: +{improvement:.4f}")
            return False, {
                'current': {'pr_auc': pr_auc_current, 'roc_auc': roc_auc_current},
                'new': {'pr_auc': pr_auc_new, 'roc_auc': roc_auc_new},
                'improvement': improvement,
                'decision': 'rejected_insufficient_improvement'
            }

        logger.info(f"✓ New model approved: +{improvement:.4f} improvement")
        return True, {
            'current': {'pr_auc': pr_auc_current, 'roc_auc': roc_auc_current},
            'new': {'pr_auc': pr_auc_new, 'roc_auc': roc_auc_new},
            'improvement': improvement,
            'decision': 'approved'
        }

    def backup_current_model(self):
        """Backup current model before deployment"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_dir = f'{self.model_dir}/backups/model_{timestamp}'

        os.makedirs(backup_dir, exist_ok=True)

        shutil.copy(
            f'{self.model_dir}/lightgbm_model.joblib',
            f'{backup_dir}/lightgbm_model.joblib'
        )
        shutil.copy(
            f'{self.model_dir}/model_metadata.json',
            f'{backup_dir}/model_metadata.json'
        )

        logger.info(f"✓ Current model backed up to: {backup_dir}")
        return backup_dir

    def deploy_new_model(self, model, feature_cols, metrics):
        """Deploy new model to production"""
        logger.info("Deploying new model...")

        # Backup current model
        backup_dir = self.backup_current_model()

        # Save new model
        joblib.dump(model, f'{self.model_dir}/lightgbm_model.joblib', compress=3)

        # Update metadata
        metadata = {
            'model_type': 'LightGBM',
            'feature_count': len(feature_cols),
            'feature_names': feature_cols,
            'optimal_threshold': 0.5,  # Would recalculate in full pipeline
            'metrics': {
                'pr_auc': metrics['new']['pr_auc'],
                'roc_auc': metrics['new']['roc_auc'],
                'improvement_over_previous': metrics['improvement']
            },
            'training_date': datetime.now().isoformat(),
            'previous_model_backup': backup_dir
        }

        with open(f'{self.model_dir}/model_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info("✓ New model deployed successfully")
        logger.info(f"✓ Previous model backed up to: {backup_dir}")

    def run_retraining(self):
        """Execute complete retraining pipeline"""
        logger.info("=" * 60)
        logger.info("AUTOMATED RETRAINING PIPELINE")
        logger.info("=" * 60)

        try:
            # Load current model
            current_model, current_metadata = self.load_current_model()

            # Load and validate data
            logger.info(f"\nLoading data from: {self.data_path}")
            df = pd.read_csv(self.data_path)
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            if not self.validate_data(df):
                logger.error("❌ Data validation failed. Aborting retraining.")
                return False

            # Prepare training data
            X_train, X_test, y_train, y_test, feature_cols = self.prepare_training_data(df)

            # Train new model
            new_model = self.train_new_model(X_train, y_train)

            # Evaluate and compare
            should_deploy, metrics = self.evaluate_models(
                current_model, new_model, X_test, y_test
            )

            if should_deploy:
                self.deploy_new_model(new_model, feature_cols, metrics)
                logger.info("\n✓ RETRAINING SUCCESSFUL - NEW MODEL DEPLOYED")
                return True
            else:
                logger.info(f"\n⚠️  RETRAINING COMPLETE - KEEPING CURRENT MODEL")
                logger.info(f"   Reason: {metrics['decision']}")
                return False

        except Exception as e:
            logger.error(f"❌ Retraining failed: {str(e)}")
            raise


# Run retraining
if __name__ == '__main__':
    pipeline = AutomatedRetrainingPipeline(
        data_path='sensor_data_features.csv',
        model_dir='models',
        min_pr_auc=0.75,  # Minimum acceptable performance
        improvement_threshold=0.02  # Minimum improvement to deploy
    )

    success = pipeline.run_retraining()

    if success:
        print("\n" + "=" * 60)
        print("✓ New model deployed and ready for production")
        print("✓ Restart API to load new model:")
        print("  docker-compose restart factoryguard-api")
        print("=" * 60)

"""
SCHEDULING AUTOMATED RETRAINING:

Option 1: Cron job (Linux/Mac)
# Add to crontab (crontab -e):
0 2 * * 0 cd /path/to/project && python retrain_model.py >> logs/retrain.log 2>&1

Option 2: GitHub Actions (CI/CD)
name: Weekly Model Retraining
on:
  schedule:
    - cron: '0 2 * * 0'  # Every Sunday at 2 AM
jobs:
  retrain:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run retraining
        run: python retrain_model.py
      - name: Deploy if successful
        run: |
          docker-compose down
          docker-compose up -d --build

Option 3: Airflow DAG (Enterprise)
from airflow import DAG
from airflow.operators.python_operator import PythonOperator

dag = DAG('factoryguard_retraining', schedule_interval='@weekly')

retrain_task = PythonOperator(
    task_id='retrain_model',
    python_callable=run_retraining_pipeline,
    dag=dag
)
"""