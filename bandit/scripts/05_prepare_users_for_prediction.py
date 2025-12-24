#!/usr/bin/env python3
"""
Phase 5 Prep: Prepare Users for Prediction

Generates a JSON file with current users and their features for Phase 5 batch predictions.

This script should run AFTER Phase 0 (which generates the Delta file) and BEFORE Phase 5 batch.

Typical flow:
1. Snowflake creates daily table → triggers Phase 0 via webhook
2. Phase 0 processes table → outputs Delta file
3. Phase 5 Prep (this script) → reads Delta file → outputs users JSON
4. Phase 5 Batch → reads users JSON → makes predictions

Usage:
  python scripts/05_prepare_users_for_prediction.py --date 2025-10-20 --delta-path dbfs:/mnt/features/daily_features.delta --output dbfs:/path/to/users.json
"""

import os
import sys
import argparse
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict

# Add bandit to path (robust to Databricks where __file__ may be undefined)
try:
    CURRENT_FILE = Path(__file__).resolve()
except NameError:
    import inspect
    frame = inspect.currentframe()
    CURRENT_FILE = Path(inspect.getfile(frame)).resolve() if frame else Path.cwd()

BANDIT_ROOT = CURRENT_FILE.parent.parent.absolute()
sys.path.insert(0, str(BANDIT_ROOT))

# Load selected features
ARTIFACTS_DIR = BANDIT_ROOT / "artifacts"
SELECTED_FEATURES_JSON = ARTIFACTS_DIR / "selected_features_50.json"  # Default, can override


def load_selected_features(features_json_path: str = None) -> List[str]:
    """Load selected features list."""
    if features_json_path:
        features_path = Path(features_json_path)
    else:
        features_path = SELECTED_FEATURES_JSON
    
    if features_path.startswith("dbfs:/"):
        features_path = Path("/dbfs" + features_path[5:])
    else:
        features_path = Path(features_path)
    
    if not features_path.exists():
        raise FileNotFoundError(f"Selected features not found: {features_path}")
    
    with open(features_path, 'r') as f:
        data = json.load(f)
    
    features = data['selected_features']
    filtered = [f for f in features if f != "action"]
    if len(filtered) != len(features):
        print("ℹ️  Dropped leakage-prone feature 'action' from selected features list")
    return filtered


def get_users_from_delta(delta_path: str, target_date: str, selected_features: List[str]) -> List[Dict]:
    """
    Extract users from Delta file for a specific date.
    
    Returns list of user contexts in Phase 5 format.
    """
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    
    spark = SparkSession.builder.appName("PrepareUsersForPrediction").getOrCreate()
    
    # Read Delta
    if delta_path.startswith("dbfs:/"):
        delta_local = "/dbfs" + delta_path[5:]
    else:
        delta_local = delta_path
    
    df = spark.read.format("delta").load(delta_local)
    
    # Filter to target date
    df = df.filter(F.col("session_date") == target_date)
    
    # Get unique users (one row per user for that date)
    # If multiple rows per user, take the latest or aggregate
    df_users = df.groupBy("user_id").agg(
        F.first("current_effectivelevelmultiplier").alias("current_effectivelevelmultiplier"),
        *[F.first(f).alias(f) for f in selected_features if f in df.columns]
    )
    
    # Convert to list of dicts
    users = []
    for row in df_users.collect():
        user = {
            "user_id": str(row["user_id"]),
            "current_effectivelevelmultiplier": float(row["current_effectivelevelmultiplier"]),
            "features": {}
        }
        
        # Extract features
        for feature in selected_features:
            if feature in row:
                user["features"][feature] = float(row[feature]) if row[feature] is not None else 0.0
            else:
                user["features"][feature] = 0.0
        
        users.append(user)
    
    return users




def main():
    parser = argparse.ArgumentParser(description="Prepare users for Phase 5 predictions")
    parser.add_argument("--date", type=str, help="Target date (YYYY-MM-DD). Defaults to today.")
    parser.add_argument("--output", type=str, required=True, help="Output JSON file path")
    parser.add_argument(
        "--delta-path",
        type=str,
        required=True,
        help="Path to Delta file from Phase 0 (e.g., dbfs:/mnt/features/daily_features.delta)"
    )
    parser.add_argument(
        "--selected-features",
        type=str,
        help="Path to selected features JSON (default: artifacts/selected_features_50.json)"
    )
    args = parser.parse_args()
    
    # Get target date
    if args.date:
        target_date = args.date
    else:
        target_date = datetime.now().strftime("%Y-%m-%d")
    
    print("=" * 80)
    print("PHASE 5 PREP: PREPARE USERS FOR PREDICTION")
    print("=" * 80)
    print(f"\n📅 Target date: {target_date}")
    print(f"📂 Delta file: {args.delta_path}")
    
    # Load selected features
    print("\n📋 Loading selected features...")
    selected_features = load_selected_features(args.selected_features)
    print(f"   Loaded {len(selected_features)} features")
    
    # Get users from Delta (output from Phase 0)
    print(f"\n👥 Fetching users from Phase 0 Delta file...")
    users = get_users_from_delta(args.delta_path, target_date, selected_features)
    
    print(f"   Found {len(users)} users")
    
    # Save to JSON
    output_path = Path(args.output)
    if str(output_path).startswith("dbfs:/"):
        output_path = Path("/dbfs" + str(output_path)[5:])
    else:
        output_path = Path(output_path)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(users, f, indent=2)
    
    print(f"\n✅ Saved {len(users)} users to {args.output}")
    print("=" * 80)
    print("READY FOR PHASE 5 BATCH PREDICTIONS")
    print("=" * 80)
    print(f"\nNext step: Run Phase 5 batch with:")
    print(f"  python scripts/05_run_fastapi_batch.py --input-file {args.output} --output-file dbfs:/path/to/predictions.json")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
