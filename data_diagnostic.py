"""
Quick diagnostic to check data quality before training/retraining
Run this if you encounter class imbalance issues
"""

import pandas as pd
import numpy as np
from datetime import timedelta

print("=" * 60)
print("DATA QUALITY DIAGNOSTIC")
print("=" * 60)

# Load data
try:
    df = pd.read_csv('sensor_data_features.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print("✓ Data loaded successfully")
except FileNotFoundError:
    print("❌ ERROR: sensor_data_features.csv not found")
    print("   Run: python 2_feature_engineering.py")
    exit(1)

print(f"\n1. DATASET SIZE")
print(f"   Total records: {len(df):,}")
print(f"   Unique robots: {df['robot_id'].nunique()}")
print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"   Duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")

print(f"\n2. CLASS DISTRIBUTION")
failures = df[df['failure_within_24h'] == 1]
normals = df[df['failure_within_24h'] == 0]

print(f"   Total failures: {len(failures):,} ({len(failures) / len(df):.2%})")
print(f"   Total normal: {len(normals):,} ({len(normals) / len(df):.2%})")

if len(failures) < 100:
    print(f"   ❌ CRITICAL: Only {len(failures)} failures - need at least 100")
    print(f"   → Re-run: python 1_generate_data.py")
elif len(failures) < 500:
    print(f"   ⚠️  WARNING: Only {len(failures)} failures - limited for robust training")
else:
    print(f"   ✓ Sufficient failures for training")

print(f"\n3. FAILURES BY ROBOT")
failure_counts = df[df['failure_within_24h'] == 1].groupby('robot_id').size()
print(f"   Robots with failures: {len(failure_counts)}")
if len(failure_counts) > 0:
    print(f"   Failures per robot (avg): {failure_counts.mean():.1f}")
    print(f"   Failures per robot (max): {failure_counts.max()}")

print(f"\n4. TEMPORAL DISTRIBUTION")
# Split data into 10 time bins
df_temp = df.copy()
df_temp['time_bin'] = pd.cut(df_temp['timestamp'], bins=10)
time_dist = df_temp.groupby('time_bin', observed=True)['failure_within_24h'].agg(['sum', 'count', 'mean'])
print(f"   Failures over time:")
for idx, row in time_dist.iterrows():
    bin_start = idx.left
    bin_end = idx.right
    print(
        f"     {bin_start.strftime('%Y-%m-%d')} to {bin_end.strftime('%Y-%m-%d')}: {int(row['sum'])} failures ({row['mean']:.2%})")

print(f"\n5. TEST SPLIT SIMULATION (for retraining)")
# Simulate the retraining split
lookback_days = 90
cutoff_date = df['timestamp'].max() - timedelta(days=lookback_days)
df_recent = df[df['timestamp'] >= cutoff_date]

print(f"   Using last {lookback_days} days for retraining")
print(f"   Recent data: {len(df_recent):,} samples")

failures_recent = df_recent[df_recent['failure_within_24h'] == 1]
normals_recent = df_recent[df_recent['failure_within_24h'] == 0]

print(f"   Recent failures: {len(failures_recent):,} ({len(failures_recent) / len(df_recent):.2%})")
print(f"   Recent normal: {len(normals_recent):,}")

# Test stratified split
test_size = 0.2
n_test_failures = max(int(len(failures_recent) * test_size), 50)
n_test_normals = int(len(normals_recent) * test_size)

print(f"\n   With 20% test split:")
print(f"     Test failures: {n_test_failures}")
print(f"     Test normal: {n_test_normals}")
print(f"     Test total: {n_test_failures + n_test_normals:,}")

if n_test_failures < 30:
    print(f"     ❌ Too few failures in test set")
    print(f"     → Need to regenerate more data")
elif n_test_failures < 50:
    print(f"     ⚠️  Limited failures in test set")
else:
    print(f"     ✓ Sufficient failures for testing")

print(f"\n6. FEATURE STATISTICS")
feature_cols = [col for col in df.columns
                if col not in ['timestamp', 'robot_id', 'failure_within_24h']]
print(f"   Total features: {len(feature_cols)}")

# Check for features with zero variance (only for numeric columns)
zero_var = []
for col in feature_cols:
    try:
        if df[col].dtype in ['float64', 'int64'] and df[col].std() == 0:
            zero_var.append(col)
    except:
        pass  # Skip non-numeric columns

if zero_var:
    print(f"   ⚠️  Zero variance features: {len(zero_var)}")
    if len(zero_var) <= 5:
        print(f"      {zero_var}")
    else:
        print(f"      {zero_var[:5]}... and {len(zero_var) - 5} more")
else:
    print(f"   ✓ All features have variance")

# Check for highly correlated features
print(f"\n7. MISSING VALUES")
missing = df[feature_cols].isnull().sum().sum()
missing_pct = missing / (len(df) * len(feature_cols))
print(f"   Total missing: {missing:,} ({missing_pct:.2%})")

if missing_pct > 0.05:
    print(f"   ⚠️  High missing rate - check feature engineering")
else:
    print(f"   ✓ Missing values acceptable")

print("\n" + "=" * 60)
print("RECOMMENDATION")
print("=" * 60)

issues = []

if len(failures) < 100:
    issues.append("CRITICAL: Too few failures")
    print("❌ CRITICAL: Too few failures in dataset")
    print("   → Re-run: python 1_generate_data.py")

if len(failures_recent) < 100:
    issues.append("CRITICAL: Too few recent failures")
    print("❌ CRITICAL: Too few failures in recent data (90 days)")
    print("   → Increase data generation period")
    print("   → Or reduce lookback_days in retraining")

if n_test_failures < 30:
    issues.append("CRITICAL: Test set too small")
    print("❌ CRITICAL: Test set would have too few failures")
    print("   → Regenerate more data")

if len(issues) == 0:
    print("✓ Dataset looks good for retraining!")
    print("  → Safe to run: python retrain_model.py")
else:
    print(f"\n⚠️  Found {len(issues)} critical issue(s)")
    print("   → Fix these before running retrain_model.py")

print("\n" + "=" * 60)
print("QUICK FIX (if needed):")
print("=" * 60)
print("""
If you have too few failures, regenerate data with higher failure rate:

1. Edit 1_generate_data.py
   Change line: self.failure_rate = 0.01
   To:          self.failure_rate = 0.02  # 2% instead of 1%

2. Regenerate data:
   python 1_generate_data.py
   python 2_feature_engineering.py

3. Retrain:
   python 3_train_model.py
   python retrain_model.py
""")
