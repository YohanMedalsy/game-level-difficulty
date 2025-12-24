#!/usr/bin/env python3
"""
Phase 1: VW Bandit Data Preparation

This script prepares data for Vowpal Wabbit contextual bandit training:
1. Load daily_features_claude.csv + daily_features_ewma_claude.csv
2. Preprocess features (one-hot encoding → 520 features)
3. Split users 80/10/10 (train/valid/test) stratified by engagement
4. Global RF feature selection on training set → 50 features
5. Validate feature stability (5 seeds, Jaccard >0.8)
6. Train propensity model (LogisticRegression)
7. Export artifacts (selected features, propensity model, scaler, splits)

Output:
- artifacts/selected_features_50.json
- models/propensity_model.pkl
- models/feature_scaler.pkl
- data/processed/train_df.parquet
- data/processed/valid_df.parquet
- data/processed/test_df.parquet
"""

import os
import sys
from pathlib import Path
import argparse
import json
import pickle
import warnings
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore")

# Add parent directory to path for imports
BANDIT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(BANDIT_ROOT / "src"))

from constants import (
    ACTION_TO_DELTA,
    ARM_ORDER,
    MIN_MULTIPLIER,
    MAX_MULTIPLIER,
    N_FEATURES_SELECTED,
    TRAIN_RATIO,
    VALID_RATIO,
    TEST_RATIO,
    get_valid_arms,
)

# Paths
DATA_RAW = BANDIT_ROOT / "data" / "raw"
DATA_PROCESSED = BANDIT_ROOT / "data" / "processed"
MODELS_DIR = BANDIT_ROOT / "models"
ARTIFACTS_DIR = BANDIT_ROOT / "artifacts"

# Create directories
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

#------------------------------------------------------------------------------
# Step 1: Load Data
#------------------------------------------------------------------------------

def load_daily_dataset(
    dataset_path: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load the combined daily dataset.

    Preferred: read from a single Delta/Parquet dataset directory produced by
    bandit_datagen_daily_pure_spark.py (contains base + advanced + EWMA).

    Fallback: merge legacy CSVs in bandit/data/raw (base + ewma).
    """
    base_path = DATA_RAW / "daily_features_claude.csv"
    ewma_path = DATA_RAW / "daily_features_ewma_claude.csv"

    print("\n" + "=" * 80)
    print("STEP 1: LOADING DATA")
    print("=" * 80)

    # Preferred path: Delta/Parquet dataset
    if dataset_path:
        ds_path = Path(dataset_path)
        if not ds_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")

        print(f"\nLoading dataset from: {dataset_path}")
        # Try native Delta first
        try:
            if (ds_path / "_delta_log").exists():
                from deltalake import DeltaTable
                print("   Detected Delta table. Reading via deltalake...")
                dt = DeltaTable(dataset_path)
                df = dt.to_pandas()
            else:
                raise Exception("Not a Delta table")
        except Exception:
            try:
                import pyarrow.dataset as ds
                print("   Reading as Parquet (pyarrow dataset)...")
                dataset = ds.dataset(dataset_path, format="parquet")
                df = dataset.to_table().to_pandas()
            except Exception as e2:
                print(f"⚠️  Failed to read dataset at {dataset_path}: {e2}")
                print("   Falling back to legacy CSV merge...")
                df = None

        if df is not None:
            # Normalize columns
            df.columns = [c.strip() for c in df.columns]
            # Ensure datetime type
            if 'session_date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['session_date']):
                df['session_date'] = pd.to_datetime(df['session_date'])
            # Apply date filters
            if start_date:
                df = df[df['session_date'] >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df['session_date'] <= pd.to_datetime(end_date)]

            print(f"  ✅ Loaded: {len(df):,} rows, {len(df.columns)} columns")

            # Filter negative rewards
            if "next_day_reward" in df.columns:
                before_filter = len(df)
                df = df[df["next_day_reward"] >= 0]
                filtered_count = before_filter - len(df)
                if filtered_count > 0:
                    print(f"\n🧹 Filtered out {filtered_count:,} rows with negative next_day_reward")

            print(f"\n✅ Final dataset: {df.shape[0]:,} rows, {df.shape[1]} columns")
            print(f"   Unique users: {df['user_id'].nunique():,}")
            return df

    # Legacy CSV merge
    if not base_path.exists():
        raise FileNotFoundError(f"Missing base CSV: {base_path}")
    if not ewma_path.exists():
        raise FileNotFoundError(f"Missing EWMA CSV: {ewma_path}")

    print(f"\nLoading base features from: {base_path}")
    base = pd.read_csv(base_path)
    base.columns = [c.strip() for c in base.columns]
    print(f"  ✅ Base: {base.shape[0]:,} rows, {base.shape[1]} columns")

    print(f"\nLoading EWMA features from: {ewma_path}")
    ewma = pd.read_csv(ewma_path)
    ewma.columns = [c.strip() for c in ewma.columns]
    print(f"  ✅ EWMA: {ewma.shape[0]:,} rows, {ewma.shape[1]} columns")

    # Merge on user_id + session_date
    key_cols = ["user_id", "session_date"]
    base_nonkeys = set(base.columns) - set(key_cols)
    ewma_unique = [c for c in ewma.columns if c not in base_nonkeys or c in key_cols]

    print(f"\nMerging on {key_cols}...")
    merged = base.merge(ewma[ewma_unique], on=key_cols, how="left")
    merged["session_date"] = pd.to_datetime(merged["session_date"])
    print(f"  ✅ Merged: {merged.shape[0]:,} rows, {merged.shape[1]} columns")

    # Filter negative rewards
    if "next_day_reward" in merged.columns:
        before_filter = len(merged)
        merged = merged[merged["next_day_reward"] >= 0]
        filtered_count = before_filter - len(merged)
        if filtered_count > 0:
            print(f"\n🧹 Filtered out {filtered_count:,} rows with negative next_day_reward")

    print(f"\n✅ Final dataset: {merged.shape[0]:,} rows, {merged.shape[1]} columns")
    print(f"   Unique users: {merged['user_id'].nunique():,}")
    return merged


#------------------------------------------------------------------------------
# Step 2: Preprocess Features
#------------------------------------------------------------------------------

def preprocess_features(
    df: pd.DataFrame,
    user_col: str = "user_id",
    arm_col: str = "difficulty_arm",
    reward_col: str = "next_day_reward",
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Preprocess features for VW training.
    Matches difficulty_ope_daily_correct.py lines 310-374

    Steps:
    1. Drop rows with missing action or reward
    2. One-hot encode categorical features
    3. Fill NaNs with 0
    4. Extract feature column list (excluding future/target columns)

    Returns:
        df_proc: Preprocessed DataFrame
        feature_cols: List of feature column names
    """
    print("\n" + "=" * 80)
    print("STEP 2: PREPROCESSING FEATURES")
    print("=" * 80)

    df_proc = df.copy()

    # Drop missing actions/rewards
    before_drop = len(df_proc)
    df_proc = df_proc.dropna(subset=[arm_col, reward_col])
    dropped = before_drop - len(df_proc)
    if dropped > 0:
        print(f"\n🧹 Dropped {dropped:,} rows with missing {arm_col} or {reward_col}")

    # Identify categorical columns
    categorical = df_proc.select_dtypes(include=["object", "category"]).columns.tolist()
    exclude_from_encoding = [
        user_col, arm_col, "session_date",
        "day_start_ts", "day_end_ts", "first_event_timestamp", "event_timestamp"
    ]
    categorical = [c for c in categorical if c not in exclude_from_encoding]

    # Drop timestamp columns
    timestamp_cols = [c for c in df_proc.columns if c in ["day_start_ts", "day_end_ts", "first_event_timestamp"]]
    if timestamp_cols:
        print(f"\n🗑️  Dropping timestamp columns: {timestamp_cols}")
        df_proc = df_proc.drop(columns=timestamp_cols)

    # One-hot encode
    if categorical:
        print(f"\n📊 One-hot encoding {len(categorical)} categorical features:")
        for cat_col in categorical:
            unique_count = df_proc[cat_col].nunique()
            print(f"  • {cat_col:20s}: {unique_count} categories")

        df_proc = pd.get_dummies(df_proc, columns=categorical, dummy_na=False)
        print(f"  ✅ One-hot encoding complete")

    # Fill NaNs
    df_proc = df_proc.fillna(0)

    # Extract feature columns (CRITICAL: No data leakage)
    exclude_cols = {
        user_col, arm_col, reward_col, "session_date",
        "next_exchangespentamountcoins", "next_day_reward", "next_effectivelevelmultiplier",
        "action"
    }
    feature_cols = [
        c for c in df_proc.columns
        if c not in exclude_cols and not c.startswith("next_")
    ]

    # Verify no leakage
    leaked = [c for c in feature_cols if c.startswith("next_")]
    if leaked:
        raise ValueError(f"DATA LEAKAGE: Future columns in features: {leaked}")

    print(f"\n✅ Preprocessing complete:")
    print(f"  • Original columns: {len(df.columns)}")
    print(f"  • After one-hot: {len(df_proc.columns)}")
    print(f"  • Feature columns: {len(feature_cols)}")

    return df_proc, feature_cols


#------------------------------------------------------------------------------
# Step 3: Split Data (User-Level Independence)
#------------------------------------------------------------------------------

def split_users_stratified(
    df: pd.DataFrame,
    user_col: str = "user_id",
    stratify_col: str = "session_count_7d",
    n_bins: int = 5,
    random_seed: int = 42,
) -> Tuple[List[str], List[str], List[str]]:
    """
    Split users into train/valid/test with stratification by engagement.

    Args:
        df: Full dataset
        user_col: User ID column
        stratify_col: Column to stratify by (default: session_count_7d)
        n_bins: Number of stratification bins
        random_seed: Random seed for reproducibility

    Returns:
        train_users, valid_users, test_users
    """
    print("\n" + "=" * 80)
    print("STEP 3: SPLITTING USERS (STRATIFIED)")
    print("=" * 80)

    # Compute user-level metric for stratification
    user_stats = df.groupby(user_col)[stratify_col].mean()
    user_engagement_quantile = pd.qcut(user_stats, q=n_bins, labels=False, duplicates='drop')

    # Convert to DataFrame
    user_strata = pd.DataFrame({
        'user_id': user_stats.index,
        'engagement_bin': user_engagement_quantile.values
    })

    print(f"\nStratification bins (by {stratify_col}):")
    print(user_strata['engagement_bin'].value_counts().sort_index())

    # Shuffle users within each stratum
    np.random.seed(random_seed)
    train_users = []
    valid_users = []
    test_users = []

    for bin_id in sorted(user_strata['engagement_bin'].unique()):
        bin_users = user_strata[user_strata['engagement_bin'] == bin_id]['user_id'].values
        np.random.shuffle(bin_users)

        n_users = len(bin_users)
        n_train = int(n_users * TRAIN_RATIO)
        n_valid = int(n_users * VALID_RATIO)

        train_users.extend(bin_users[:n_train])
        valid_users.extend(bin_users[n_train:n_train + n_valid])
        test_users.extend(bin_users[n_train + n_valid:])

    print(f"\n✅ User split:")
    print(f"  • Train: {len(train_users):,} users ({len(train_users)/len(user_stats)*100:.1f}%)")
    print(f"  • Valid: {len(valid_users):,} users ({len(valid_users)/len(user_stats)*100:.1f}%)")
    print(f"  • Test:  {len(test_users):,} users ({len(test_users)/len(user_stats)*100:.1f}%)")

    # Verify no overlap
    assert len(set(train_users) & set(valid_users)) == 0, "Train/valid overlap!"
    assert len(set(train_users) & set(test_users)) == 0, "Train/test overlap!"
    assert len(set(valid_users) & set(test_users)) == 0, "Valid/test overlap!"

    return train_users, valid_users, test_users


#------------------------------------------------------------------------------
# Step 4: RF Feature Selection (Matching OPE)
#------------------------------------------------------------------------------

def rf_prescreen_correct(
    df: pd.DataFrame,
    features: List[str],
    reward_col: str,
    arm_col: str,
    top_n: int = N_FEATURES_SELECTED,
    sample_frac: float = 0.8,
    random_seed: int = 42,
) -> List[str]:
    """
    RF-based feature selection using arm-weighted importance.

    CRITICAL FIX: Uses arm prediction instead of reward (no target leakage).
    Matches difficulty_ope_daily_correct.py lines 380-409

    Args:
        df: Training dataframe
        features: List of all feature names
        reward_col: Reward column (not used, just for signature)
        arm_col: Arm/action column
        top_n: Number of features to select
        sample_frac: Fraction of data to sample for speed
        random_seed: Random seed

    Returns:
        List of selected feature names
    """
    print(f"\n🌲 Running RF feature selection (top {top_n} from {len(features)})...")

    # Sample for speed
    if sample_frac < 1.0:
        df_sample = df.sample(frac=sample_frac, random_state=random_seed)
        print(f"  • Sampled {len(df_sample):,} / {len(df):,} rows ({sample_frac*100:.0f}%)")
    else:
        df_sample = df

    # Prepare data
    X = df_sample[features].values
    y_arm = df_sample[arm_col].astype('category').cat.codes.values  # Convert arms to numeric

    # Train RF on arm prediction (NOT reward - avoids leakage)
    print(f"  • Training RF classifier on {len(np.unique(y_arm))} arms...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=random_seed,
        n_jobs=-1,
        verbose=0
    )
    rf.fit(X, y_arm)

    # Get feature importances
    importances = rf.feature_importances_
    feature_importance = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)

    selected_features = [f[0] for f in feature_importance[:top_n]]

    print(f"  ✅ Selected top {len(selected_features)} features")
    print(f"\n  Top 10 features:")
    for i, (feature, importance) in enumerate(feature_importance[:10], 1):
        print(f"    {i:2d}. {feature:40s} {importance:.4f}")

    return selected_features


def validate_feature_stability(
    df: pd.DataFrame,
    features: List[str],
    arm_col: str,
    top_n: int = N_FEATURES_SELECTED,
    n_seeds: int = 5,
    jaccard_threshold: float = 0.8,
) -> float:
    """
    Validate feature selection stability across multiple random seeds.

    Args:
        df: Training dataframe
        features: Full feature list
        arm_col: Arm column
        top_n: Number of features to select
        n_seeds: Number of seeds to test
        jaccard_threshold: Minimum acceptable Jaccard similarity

    Returns:
        Mean Jaccard similarity
    """
    print(f"\n🔍 Validating feature stability ({n_seeds} seeds, threshold={jaccard_threshold})...")

    selected_feature_sets = []
    for seed in range(42, 42 + n_seeds):
        selected = rf_prescreen_correct(
            df, features, reward_col="next_day_reward", arm_col=arm_col,
            top_n=top_n, sample_frac=0.8, random_seed=seed
        )
        selected_feature_sets.append(set(selected))

    # Compute pairwise Jaccard similarities
    jaccard_scores = []
    for i in range(len(selected_feature_sets)):
        for j in range(i + 1, len(selected_feature_sets)):
            intersection = len(selected_feature_sets[i] & selected_feature_sets[j])
            union = len(selected_feature_sets[i] | selected_feature_sets[j])
            jaccard = intersection / union if union > 0 else 0
            jaccard_scores.append(jaccard)

    mean_jaccard = np.mean(jaccard_scores)
    print(f"\n  ✅ Jaccard similarity: {mean_jaccard:.3f} (threshold: {jaccard_threshold})")

    if mean_jaccard < jaccard_threshold:
        print(f"  ⚠️  WARNING: Feature selection unstable! Consider increasing sample_frac or top_n")
    else:
        print(f"  ✅ Feature selection is stable!")

    return mean_jaccard


#------------------------------------------------------------------------------
# Step 5: Train Propensity Model
#------------------------------------------------------------------------------

def train_propensity_model(
    df_train: pd.DataFrame,
    features: List[str],
    arm_col: str = "difficulty_arm",
    random_seed: int = 42,
) -> Pipeline:
    """
    Train propensity model (LogisticRegression) for logging probabilities.

    Matches difficulty_ope_daily_correct.py lines 456-465

    Args:
        df_train: Training dataframe
        features: Selected feature names (50 features)
        arm_col: Arm column
        random_seed: Random seed

    Returns:
        Trained propensity model pipeline
    """
    print("\n" + "=" * 80)
    print("STEP 5: TRAINING PROPENSITY MODEL")
    print("=" * 80)

    X_train = df_train[features].values
    y_train = df_train[arm_col].values

    print(f"\nTraining LogisticRegression on {X_train.shape[0]:,} samples, {X_train.shape[1]} features...")

    propensity_model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=500,
            multi_class="multinomial",
            solver="lbfgs",
            class_weight="balanced",
            random_state=random_seed,
            n_jobs=-1,
            verbose=0
        ))
    ])

    propensity_model.fit(X_train, y_train)

    # Evaluate
    y_pred = propensity_model.predict(X_train)
    accuracy = (y_pred == y_train).mean()
    print(f"\n  ✅ Training accuracy: {accuracy*100:.2f}%")

    # Show classification report
    print(f"\n  Classification report:")
    report = classification_report(y_train, y_pred, target_names=ARM_ORDER, zero_division=0)
    for line in report.split('\n'):
        if line.strip():
            print(f"    {line}")

    return propensity_model


#------------------------------------------------------------------------------
# Main Pipeline
#------------------------------------------------------------------------------

def main():
    """Run full data preparation pipeline."""

    print("\n" + "="*80)
    print(" " * 20 + "VW BANDIT DATA PREPARATION")
    print(" " * 25 + "PHASE 1")
    print("="*80)

    # CLI overrides for dataset path and date filters
    dataset_path = os.environ.get('BANDIT_DATASET_PATH', '')
    start_date = os.environ.get('BANDIT_START_DATE', '') or None
    end_date = os.environ.get('BANDIT_END_DATE', '') or None

    # CLI flags (override env if set)
    parser = argparse.ArgumentParser(description="Phase 1: VW Bandit Data Preparation")
    parser.add_argument('--dataset-path', type=str, default=os.environ.get('BANDIT_DATASET_PATH', ''), help='Path to Delta/Parquet dataset directory (from datagen)')
    parser.add_argument('--start-date', type=str, default=os.environ.get('BANDIT_START_DATE', '2025-10-01'), help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=os.environ.get('BANDIT_END_DATE', ''), help='End date (YYYY-MM-DD)')
    args = parser.parse_args()

    dataset_path = args.dataset_path if args.dataset_path else None
    start_date = args.start_date if args.start_date else None
    end_date = args.end_date if args.end_date else None

    # Step 1: Load data
    df = load_daily_dataset(dataset_path=dataset_path,
                            start_date=start_date, end_date=end_date)

    # Step 2: Preprocess features
    df_proc, all_features = preprocess_features(df)
    print(f"\n  → {len(all_features)} features after preprocessing")

    # Step 3: Split users
    train_users, valid_users, test_users = split_users_stratified(
        df_proc, stratify_col="session_count_7d", n_bins=5
    )

    train_df = df_proc[df_proc['user_id'].isin(train_users)].copy()
    valid_df = df_proc[df_proc['user_id'].isin(valid_users)].copy()
    test_df = df_proc[df_proc['user_id'].isin(test_users)].copy()

    print(f"\n  Dataset sizes:")
    print(f"    • Train: {len(train_df):,} rows ({len(train_df)/len(df_proc)*100:.1f}%)")
    print(f"    • Valid: {len(valid_df):,} rows ({len(valid_df)/len(df_proc)*100:.1f}%)")
    print(f"    • Test:  {len(test_df):,} rows ({len(test_df)/len(df_proc)*100:.1f}%)")

    # Step 4: Feature selection on training set ONLY
    print("\n" + "=" * 80)
    print("STEP 4: FEATURE SELECTION")
    print("=" * 80)

    selected_features = rf_prescreen_correct(
        train_df, all_features,
        reward_col="next_day_reward",
        arm_col="difficulty_arm",
        top_n=N_FEATURES_SELECTED,
        sample_frac=0.8
    )

    # Validate stability
    jaccard_score = validate_feature_stability(
        train_df, all_features, arm_col="difficulty_arm",
        top_n=N_FEATURES_SELECTED, n_seeds=5, jaccard_threshold=0.8
    )

    # Step 5: Train propensity model on selected features
    propensity_model = train_propensity_model(
        train_df, selected_features, arm_col="difficulty_arm"
    )

    # Step 6: Export artifacts
    print("\n" + "=" * 80)
    print("STEP 6: EXPORTING ARTIFACTS")
    print("=" * 80)

    # Save selected features
    features_path = ARTIFACTS_DIR / "selected_features_50.json"
    with open(features_path, 'w') as f:
        json.dump({
            'selected_features': selected_features,
            'n_features': len(selected_features),
            'selection_method': 'rf_arm_weighted',
            'jaccard_stability': float(jaccard_score),
        }, f, indent=2)
    print(f"\n✅ Saved selected features: {features_path}")

    # Save propensity model
    propensity_path = MODELS_DIR / "propensity_model.pkl"
    with open(propensity_path, 'wb') as f:
        pickle.dump(propensity_model, f)
    print(f"✅ Saved propensity model: {propensity_path}")

    # Save feature scaler separately (for inference)
    scaler_path = MODELS_DIR / "feature_scaler.pkl"
    scaler = propensity_model.named_steps['scaler']
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✅ Saved feature scaler: {scaler_path}")

    # Save data splits (only keep selected features + metadata)
    metadata_cols = ['user_id', 'session_date', 'difficulty_arm', 'next_day_reward',
                     'current_effectivelevelmultiplier', 'next_effectivelevelmultiplier']
    keep_cols = metadata_cols + selected_features

    train_output = DATA_PROCESSED / "train_df.parquet"
    train_df[keep_cols].to_parquet(train_output, index=False)
    print(f"✅ Saved train split: {train_output} ({len(train_df):,} rows)")

    valid_output = DATA_PROCESSED / "valid_df.parquet"
    valid_df[keep_cols].to_parquet(valid_output, index=False)
    print(f"✅ Saved valid split: {valid_output} ({len(valid_df):,} rows)")

    test_output = DATA_PROCESSED / "test_df.parquet"
    test_df[keep_cols].to_parquet(test_output, index=False)
    print(f"✅ Saved test split: {test_output} ({len(test_df):,} rows)")

    # Summary
    print("\n" + "=" * 80)
    print("DATA PREPARATION COMPLETE!")
    print("=" * 80)
    print(f"\n✅ Artifacts exported:")
    print(f"  • Selected features (50):  {features_path}")
    print(f"  • Propensity model:        {propensity_path}")
    print(f"  • Feature scaler:          {scaler_path}")
    print(f"  • Train split:             {train_output}")
    print(f"  • Valid split:             {valid_output}")
    print(f"  • Test split:              {test_output}")
    print(f"\n🎯 Next step: Run 02_convert_to_vw.py to create VW CB_ADF format files")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
