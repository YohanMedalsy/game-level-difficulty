#!/usr/bin/env python3
"""
Phase 2: Convert Data to VW CB_ADF Format

Converts preprocessed Parquet files to Vowpal Wabbit Contextual Bandit Action-Dependent Features format.

VW CB_ADF Format:
shared |u user_lifetime:45 |s session_count:12 |e coins:1500
0:-500:0.18 |a arm:0 delta:-0.12 |c mult:0.88
1:-350:0.22 |a arm:1 delta:-0.06 |c mult:0.88
2:-280:0.35 |a arm:2 delta:0.00 |c mult:0.88
3:-420:0.15 |a arm:3 delta:0.06 |c mult:0.88
4:-610:0.10 |a arm:4 delta:0.12 |c mult:0.88

Format: action:cost:probability |namespace feature:value ...
- action: Arm index (0-4)
- cost: Negative reward (VW minimizes cost)
- probability: Propensity from behavior policy
- Namespaces: u=user, s=session, e=economy, t=temporal, l=lag, w=ewma, a=action, c=context

Output:
- data/processed/train.vw
- data/processed/valid.vw
- data/processed/test.vw
"""

import sys
from pathlib import Path
import json
import pickle
from typing import List, Dict, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
BANDIT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(BANDIT_ROOT / "src"))

from constants import (
    ACTION_TO_DELTA,
    ARM_ORDER,
    N_ARMS,
    assign_feature_to_namespace,
    get_valid_arms,
)

# Paths
DATA_PROCESSED = BANDIT_ROOT / "data" / "processed"
MODELS_DIR = BANDIT_ROOT / "models"
ARTIFACTS_DIR = BANDIT_ROOT / "artifacts"


def load_artifacts():
    """Load selected features and propensity model."""
    print("\n" + "=" * 80)
    print("LOADING ARTIFACTS FROM PHASE 1")
    print("=" * 80)

    # Load selected features
    features_path = ARTIFACTS_DIR / "selected_features_50.json"
    if not features_path.exists():
        raise FileNotFoundError(f"Run Phase 1 first! Missing: {features_path}")

    with open(features_path, 'r') as f:
        feature_data = json.load(f)
    selected_features = feature_data['selected_features']
    cleaned_features = [f for f in selected_features if f != "action"]
    if len(cleaned_features) != len(selected_features):
        print("ℹ️  Dropped leakage-prone feature 'action' from selected features list")
    selected_features = cleaned_features
    print(f"\n✅ Loaded {len(selected_features)} selected features")

    # Load propensity model
    propensity_path = MODELS_DIR / "propensity_model.pkl"
    if not propensity_path.exists():
        raise FileNotFoundError(f"Run Phase 1 first! Missing: {propensity_path}")

    with open(propensity_path, 'rb') as f:
        propensity_model = pickle.load(f)
    print(f"✅ Loaded propensity model")

    return selected_features, propensity_model


def organize_features_by_namespace(features: List[str]) -> Dict[str, List[str]]:
    """
    Organize features into VW namespaces based on naming patterns.

    Returns:
        Dict mapping namespace abbreviation to list of features
    """
    namespace_features = defaultdict(list)

    for feature in features:
        namespace = assign_feature_to_namespace(feature)
        namespace_features[namespace].append(feature)

    print(f"\n📋 Feature organization by namespace:")
    for ns, feats in sorted(namespace_features.items()):
        print(f"  |{ns} : {len(feats)} features")
        if len(feats) <= 5:
            for f in feats:
                print(f"      - {f}")
        else:
            for f in feats[:3]:
                print(f"      - {f}")
            print(f"      ... ({len(feats) - 3} more)")

    return dict(namespace_features)


def format_vw_features(
    feature_values: pd.Series,
    namespace_features: Dict[str, List[str]]
) -> str:
    """
    Format features into VW namespace format.

    Args:
        feature_values: Series with feature values
        namespace_features: Dict mapping namespace to feature names

    Returns:
        String like "|u user_lifetime:45 session_count:12 |s coins:1500"
    """
    parts = []

    for namespace, features in sorted(namespace_features.items()):
        # Get values for this namespace
        ns_parts = []
        for feature in features:
            value = feature_values.get(feature, 0)
            if value != 0:  # VW ignores 0 values, so skip for efficiency
                # Clean feature name (VW doesn't like certain characters)
                clean_name = feature.replace(' ', '_').replace(':', '_').replace('|', '_')
                ns_parts.append(f"{clean_name}:{value:.6f}")

        if ns_parts:
            parts.append(f"|{namespace} " + " ".join(ns_parts))

    return " ".join(parts)


def convert_row_to_vw_adf(
    row: pd.Series,
    selected_features: List[str],
    namespace_features: Dict[str, List[str]],
    propensity_model,
    arm_col: str = "difficulty_arm",
    reward_col: str = "next_day_reward",
) -> str:
    """
    Convert a single row to VW CB_ADF format.

    Returns:
        Multi-line string representing one VW example (shared + N_ARMS action lines)
    """
    # Extract features for shared context
    feature_values = row[selected_features]
    shared_line = "shared " + format_vw_features(feature_values, namespace_features)

    # Get propensity probabilities for all arms
    X_context = feature_values.values.reshape(1, -1)
    propensities = propensity_model.predict_proba(X_context)[0]  # Shape: (N_ARMS,)

    # Map arms to propensities
    arm_classes = propensity_model.classes_
    arm_to_prob = {arm: propensities[i] for i, arm in enumerate(arm_classes)}

    # Ensure all arms have probabilities (fill missing with small value)
    for arm in ARM_ORDER:
        if arm not in arm_to_prob:
            arm_to_prob[arm] = 1e-6

    # Renormalize to sum to 1.0
    total_prob = sum(arm_to_prob.values())
    arm_to_prob = {arm: prob / total_prob for arm, prob in arm_to_prob.items()}

    # Get observed arm and reward
    observed_arm = row[arm_col]
    reward = row[reward_col]
    cost = -reward  # VW minimizes cost

    # Get current multiplier for feasibility
    current_mult = row['current_effectivelevelmultiplier']
    valid_arms = get_valid_arms(current_mult)

    # Create action lines
    action_lines = []
    for arm_idx, arm in enumerate(ARM_ORDER):
        delta = ACTION_TO_DELTA[arm]
        feasible = 1 if arm in valid_arms else 0
        action_features = f"|a arm:{arm_idx} delta:{delta:.2f} |c mult:{current_mult:.2f} feasible:{feasible}"

        if arm == observed_arm:
            action_cost = cost
            action_prob = arm_to_prob[arm]
            label_prefix = f"{arm_idx}:{action_cost:.4f}:{action_prob:.6f} "
        else:
            label_prefix = ""

        action_line = f"{label_prefix}{action_features}"
        action_lines.append(action_line)

    # Combine shared + actions
    vw_example = shared_line + "\n" + "\n".join(action_lines)
    return vw_example


def convert_dataset_to_vw(
    df: pd.DataFrame,
    selected_features: List[str],
    namespace_features: Dict[str, List[str]],
    propensity_model,
    output_path: Path,
    arm_col: str = "difficulty_arm",
    reward_col: str = "next_day_reward",
):
    """
    Convert full dataset to VW format and write to file.

    Args:
        df: Dataset to convert
        selected_features: List of feature names
        namespace_features: Feature organization by namespace
        propensity_model: Trained propensity model
        output_path: Where to write VW file
        arm_col: Arm column name
        reward_col: Reward column name
    """
    print(f"\n🔄 Converting {len(df):,} rows to VW format...")
    print(f"   Output: {output_path}")

    with open(output_path, 'w') as f:
        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Converting"):
            vw_example = convert_row_to_vw_adf(
                row, selected_features, namespace_features,
                propensity_model, arm_col, reward_col
            )
            f.write(vw_example + "\n\n")  # Blank line between examples

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"   ✅ Wrote {len(df):,} examples ({file_size_mb:.1f} MB)")


def validate_vw_format(vw_path: Path, n_examples: int = 5):
    """
    Validate VW file format by showing first few examples.

    Args:
        vw_path: Path to VW file
        n_examples: Number of examples to display
    """
    print(f"\n🔍 Validating VW format: {vw_path.name}")

    with open(vw_path, 'r') as f:
        content = f.read()

    examples = content.strip().split('\n\n')
    print(f"   Total examples: {len(examples):,}")

    print(f"\n   First {min(n_examples, len(examples))} examples:")
    for i, example in enumerate(examples[:n_examples], 1):
        lines = example.strip().split('\n')
        print(f"\n   Example {i}:")
        for j, line in enumerate(lines):
            if j == 0:
                # Shared line - truncate if too long
                if len(line) > 100:
                    print(f"      {line[:100]}...")
                else:
                    print(f"      {line}")
            else:
                # Action line - show first 80 chars
                if len(line) > 80:
                    print(f"      {line[:80]}...")
                else:
                    print(f"      {line}")
        if i < n_examples:
            print("      " + "-" * 40)


def main():
    """Run VW data conversion pipeline."""

    print("\n" + "=" * 80)
    print(" " * 20 + "VW DATA CONVERSION TO CB_ADF FORMAT")
    print(" " * 30 + "PHASE 2")
    print("=" * 80)

    # Load artifacts from Phase 1
    selected_features, propensity_model = load_artifacts()

    # Organize features by namespace
    namespace_features = organize_features_by_namespace(selected_features)

    # Load preprocessed data splits
    print("\n" + "=" * 80)
    print("LOADING PREPROCESSED DATA")
    print("=" * 80)

    train_path = DATA_PROCESSED / "train_df.parquet"
    valid_path = DATA_PROCESSED / "valid_df.parquet"
    test_path = DATA_PROCESSED / "test_df.parquet"

    if not train_path.exists():
        raise FileNotFoundError(f"Run Phase 1 first! Missing: {train_path}")

    train_df = pd.read_parquet(train_path)
    valid_df = pd.read_parquet(valid_path)
    test_df = pd.read_parquet(test_path)

    print(f"\n✅ Loaded data splits:")
    print(f"   Train: {len(train_df):,} rows")
    print(f"   Valid: {len(valid_df):,} rows")
    print(f"   Test:  {len(test_df):,} rows")

    # Convert to VW format
    print("\n" + "=" * 80)
    print("CONVERTING TO VW CB_ADF FORMAT")
    print("=" * 80)

    # Train
    train_vw_path = DATA_PROCESSED / "train.vw"
    convert_dataset_to_vw(
        train_df, selected_features, namespace_features,
        propensity_model, train_vw_path
    )

    # Valid
    valid_vw_path = DATA_PROCESSED / "valid.vw"
    convert_dataset_to_vw(
        valid_df, selected_features, namespace_features,
        propensity_model, valid_vw_path
    )

    # Test
    test_vw_path = DATA_PROCESSED / "test.vw"
    convert_dataset_to_vw(
        test_df, selected_features, namespace_features,
        propensity_model, test_vw_path
    )

    # Validate format
    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)

    validate_vw_format(train_vw_path, n_examples=3)

    # Summary
    print("\n" + "=" * 80)
    print("VW DATA CONVERSION COMPLETE!")
    print("=" * 80)
    print(f"\n✅ VW files created:")
    print(f"   Train: {train_vw_path}")
    print(f"   Valid: {valid_vw_path}")
    print(f"   Test:  {test_vw_path}")
    print(f"\n🎯 Next step: Run 03_train_vw_optuna.py to train VW model with hyperparameter optimization")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
