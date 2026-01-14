"""
Quick fix to update optimal threshold in model metadata
Run this if threshold is incorrectly set to 1.0
"""

import json
import joblib
import pandas as pd
from sklearn.metrics import precision_recall_curve

print("=" * 60)
print("FIXING OPTIMAL THRESHOLD")
print("=" * 60)

# Load model and metadata
print("\n1. Loading model...")
model = joblib.load('models/lightgbm_model.joblib')

with open('models/model_metadata.json', 'r') as f:
    metadata = json.load(f)

print(f"   Current threshold: {metadata['optimal_threshold']}")

# Load test data
print("\n2. Loading data to recalculate threshold...")
df = pd.read_csv('sensor_data_features.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Use same split strategy as training
df = df.sort_values('timestamp')
failures = df[df['failure_within_24h'] == 1]
normals = df[df['failure_within_24h'] == 0]

test_size = 0.2
n_test_failures = max(int(len(failures) * test_size), 50)
n_test_normals = int(len(normals) * test_size)

test_failures = failures.tail(n_test_failures)
test_normals = normals.tail(n_test_normals)
test_df = pd.concat([test_failures, test_normals]).sort_values('timestamp')

# Get features
feature_cols = metadata['feature_names']
X_test = test_df[feature_cols]
y_test = test_df['failure_within_24h']

print(f"   Test set: {len(X_test):,} samples ({y_test.mean():.2%} failure)")

# Predict probabilities
print("\n3. Calculating optimal threshold...")
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Cost-based optimization
fp_cost = 5000  # False positive cost
fn_cost = 2000000  # False negative cost

precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

min_cost = float('inf')
optimal_threshold = 0.5

for threshold in thresholds:
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Calculate costs
    fp = ((y_pred == 1) & (y_test == 0)).sum()
    fn = ((y_pred == 0) & (y_test == 1)).sum()

    cost = (fp * fp_cost) + (fn * fn_cost)

    if cost < min_cost:
        min_cost = cost
        optimal_threshold = threshold

print(f"   ✓ Optimal threshold: {optimal_threshold:.4f}")
print(f"   ✓ Minimum cost: ${min_cost:,.0f}")

# Test predictions at optimal threshold
y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
tp = ((y_pred_optimal == 1) & (y_test == 1)).sum()
fp = ((y_pred_optimal == 1) & (y_test == 0)).sum()
tn = ((y_pred_optimal == 0) & (y_test == 0)).sum()
fn = ((y_pred_optimal == 0) & (y_test == 1)).sum()

print(f"\n4. Performance at optimal threshold:")
print(f"   True Positives:  {tp} (caught failures)")
print(f"   False Positives: {fp} (false alarms)")
print(f"   True Negatives:  {tn} (correct normal)")
print(f"   False Negatives: {fn} (missed failures)")

recall_score = tp / (tp + fn) if (tp + fn) > 0 else 0
precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0

print(f"\n   Recall: {recall_score:.2%} (% of failures caught)")
print(f"   Precision: {precision_score:.2%} (% of alarms correct)")

# Update metadata
print("\n5. Updating metadata...")
metadata['optimal_threshold'] = float(optimal_threshold)
metadata['threshold_optimization'] = {
    'method': 'cost-based',
    'fp_cost': fp_cost,
    'fn_cost': fn_cost,
    'min_cost': float(min_cost),
    'recall': float(recall_score),
    'precision': float(precision_score)
}

with open('models/model_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)

print("   ✓ Metadata updated")

print("\n" + "=" * 60)
print("✓ THRESHOLD FIXED SUCCESSFULLY")
print("=" * 60)
print("\nNext steps:")
print("1. Restart API: python app.py")
print("2. Test predictions: python api_client_demo.py")
print("\nThe model will now correctly flag high-risk robots!")