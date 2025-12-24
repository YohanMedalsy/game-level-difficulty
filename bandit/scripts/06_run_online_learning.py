#!/usr/bin/env python3
"""
Phase 6: Daily Online Learning Cycle

Updates VW model daily with same-day reward feedback.

Daily Cycle (End of Day T):
  6:00am - Aggregate Day T-1 contexts + actions + Day T rewards
  7:00am - Incremental VW update with new data
  8:00am - Deploy updated model (atomic swap)
  9:00am - Ready for Day T+1 predictions

Features:
- Same-day reward feedback (no delay)
- Incremental training (--initial_regressor)
- Atomic model deployment (no downtime)
- Performance monitoring (daily DR estimate)
- Rollback mechanism (revert to previous day)
- Propensity drift detection

Usage:
  python scripts/06_run_online_learning.py --date 2025-10-20
  python scripts/06_run_online_learning.py  # Uses yesterday
"""

import sys
import os
import subprocess
import argparse
import glob
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import json
import pickle
import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm

# Add parent to path - handle Databricks environment where __file__ is not defined
try:
    BANDIT_ROOT = Path(__file__).parent.parent.absolute()
except NameError:
    # Databricks: use workspace path or fallback
    import os as _os
    # Try to detect from Databricks workspace path
    workspace_path = _os.environ.get("DATABRICKS_WORKSPACE_PATH", "/Workspace/Users/yohan.medalsy@spaceplay.games/ai/bandit")
    if _os.path.exists(workspace_path):
        BANDIT_ROOT = Path(workspace_path)
    else:
        # Fallback to current working directory
        BANDIT_ROOT = Path(_os.getcwd())
        if not (BANDIT_ROOT / "src").exists():
            BANDIT_ROOT = BANDIT_ROOT / "bandit"

sys.path.insert(0, str(BANDIT_ROOT / "src"))

from constants import ARM_ORDER, get_valid_arms, ACTION_TO_DELTA
from validation.ope_estimators import ope_ips_snips_dr, compute_uniform_target_policy

# Paths
# PRODUCTION: Use DBFS mounts for persistent storage of models and logs
MODELS_DIR = Path("/dbfs/mnt/vw_pipeline/models")
ARTIFACTS_DIR = BANDIT_ROOT / "artifacts"
# Logs must persist outside the ephemeral job cluster
LOGS_DIR = Path("/dbfs/mnt/bandit/logs/online_learning") 
DATA_PROCESSED = BANDIT_ROOT / "data" / "processed"
INFERENCE_LOGS = Path("/dbfs/mnt/bandit/logs/inference")

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def get_latest_model(models_dir: Path = MODELS_DIR, pattern: str = "vw_bandit_dr_*.vw") -> Path:
    """
    Get the latest timestamped VW model.
    
    Args:
        models_dir: Directory containing models
        pattern: Glob pattern for model files
        
    Returns:
        Path to latest model
        
    Raises:
        FileNotFoundError: If no models found
    """
    # Convert DBFS path to local if needed
    if str(models_dir).startswith('dbfs:/'):
        models_dir = Path(str(models_dir).replace('dbfs:/', '/dbfs/'))
    
    # Find all matching models
    models = list(models_dir.glob(pattern))
    
    if not models:
        raise FileNotFoundError(f"No models found matching {pattern} in {models_dir}")
    
    # Sort by modification time (newest first)
    models.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    
    return models[0]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Daily online learning update")
    parser.add_argument(
        '--date',
        type=str,
        help='Date to process (YYYY-MM-DD). Defaults to yesterday.'
    )
    parser.add_argument(
        '--selected-features',
        type=str,
        required=True,
        help='Path to selected features JSON (must match Phase 2/5)'
    )
    parser.add_argument(
        '--delta-path',
        type=str,
        required=True,
        help='Path to Phase 0 Delta table (for reward fetching and Phase 5 trigger)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Dry run without deploying model'
    )
    parser.add_argument(
        '--trigger-phase5',
        action='store_true',
        help='Trigger Phase 5 batch inference after model update'
    )
    parser.add_argument(
        '--phase5-job-id',
        type=int,
        default=899063372642667,
        help='Databricks Job ID for Phase 5 (Cloudflare KV version)'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        help='Specific model path to use as base for incremental learning (overrides automatic latest detection)'
    )
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip model training - only discover rewards and update decisions table. Use this until model is being served in production.'
    )
    return parser.parse_args()


def get_target_date(date_str: str = None) -> datetime:
    """Get target date for online learning."""
    if date_str:
        return datetime.strptime(date_str, '%Y-%m-%d')
    else:
        # Default to yesterday
        return datetime.now() - timedelta(days=1)


def ensure_decisions_schema(spark, decisions_table: str) -> None:
    """
    Ensure the decisions table has reward columns.
    Adds reward_amount and reward_session_date if they don't exist.
    """
    print(f"\n🔧 Checking decisions table schema...")
    
    try:
        # Get current schema
        schema = spark.table(decisions_table).schema
        existing_cols = {f.name.lower() for f in schema.fields}
        
        # Add reward_amount if missing
        if 'reward_amount' not in existing_cols:
            print(f"  Adding reward_amount column...")
            spark.sql(f"ALTER TABLE {decisions_table} ADD COLUMN reward_amount DOUBLE")
            print(f"  ✅ Added reward_amount column")
        
        # Add reward_session_date if missing
        if 'reward_session_date' not in existing_cols:
            print(f"  Adding reward_session_date column...")
            spark.sql(f"ALTER TABLE {decisions_table} ADD COLUMN reward_session_date STRING")
            print(f"  ✅ Added reward_session_date column")
        
        if 'reward_amount' in existing_cols and 'reward_session_date' in existing_cols:
            print(f"  ✅ Schema already has reward columns")
            
    except Exception as e:
        print(f"  ⚠️  Could not modify schema: {e}")
        print(f"  Continuing anyway - columns may already exist")


def aggregate_decisions_and_rewards(
    target_date: datetime,
    decisions_table: str,
    delta_path: str,
    selected_features: List[str]
) -> 'DataFrame':
    """
    Aggregate decisions and fetch rewards using pure Spark.
    
    CORRECTED LOGIC:
    - Loads ALL decisions without rewards (reward_amount IS NULL)
    - Finds FIRST session AFTER decision_date for each user
    - Uses current_exchangespentamountcoins from that session as reward
    - This handles users who skip days between decision and next play
    
    Args:
        target_date: Date to process (used for logging only)
        decisions_table: SQL table with logged decisions
        delta_path: Path to Phase 0 Delta table
        selected_features: List of feature names
        
    Returns:
        Spark DataFrame with decisions + rewards (only decisions that now have rewards)
    """
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType, MapType
    
    print("\n" + "=" * 80)
    print(f"STEP 1: AGGREGATING DECISIONS & REWARDS")
    print(f"        (Finding rewards for ALL pending decisions)")
    print("=" * 80)
    
    spark = SparkSession.builder.appName("Phase6OnlineLearning").getOrCreate()
    date_str = target_date.strftime('%Y-%m-%d')
    
    # Ensure schema has reward columns
    ensure_decisions_schema(spark, decisions_table)
    
    # Step 1: Read ALL decisions WITHOUT rewards
    print(f"\n📥 Reading pending decisions from: {decisions_table}")
    print(f"   Filter: decision_date < '{date_str}' AND reward_amount IS NULL")
    
    try:
        # Load decisions that:
        # 1. Were made BEFORE today (not today's decisions - those are new)
        # 2. Don't have a reward yet (reward_amount IS NULL)
        decisions_df = spark.table(decisions_table).filter(
            f"decision_date < '{date_str}' AND reward_amount IS NULL"
        )
        
        decision_count = decisions_df.count()
        if decision_count == 0:
            print(f"  ⚠️  No pending decisions found")
            return spark.createDataFrame([], schema=StructType([]))
        
        # Show breakdown by decision_date
        date_breakdown = decisions_df.groupBy("decision_date").count().orderBy("decision_date").collect()
        print(f"  ✅ Loaded {decision_count:,} pending decisions:")
        for row in date_breakdown:
            print(f"      {row['decision_date']}: {row['count']:,} decisions")
        
    except Exception as e:
        print(f"  ❌ Failed to read from table {decisions_table}: {e}")
        raise e
    
    # Step 2: Read ALL historical Delta data (for reward lookup)
    print(f"\n📥 Reading Phase 0 Delta table (all sessions): {delta_path}")
    
    try:
        # Normalize path for Spark
        if delta_path.startswith("dbfs:/"):
            spark_delta_path = delta_path  # Use dbfs:/ path for Spark
        elif delta_path.startswith("/dbfs/"):
            spark_delta_path = "dbfs:" + delta_path[5:]  # Convert to dbfs:/
        else:
            spark_delta_path = delta_path
        
        delta_df = spark.read.format("delta").load(spark_delta_path)
        
        delta_count = delta_df.count()
        delta_dates = delta_df.select("session_date").distinct().orderBy("session_date").collect()
        print(f"  ✅ Loaded {delta_count:,} rows from Delta")
        print(f"  📅 Session dates available: {[str(r[0]) for r in delta_dates]}")
        
        # Check for required columns - validate ALL required columns explicitly
        delta_cols = {c.lower() for c in delta_df.columns}
        
        # Check for join columns first (these are critical)
        if 'user_id' not in delta_cols:
            raise ValueError(f"Delta table missing required column 'user_id'. Available: {list(delta_df.columns)}")
        if 'session_date' not in delta_cols:
            raise ValueError(f"Delta table missing required column 'session_date'. Available: {list(delta_df.columns)}")
        
        # Check for reward column with fallback
        if 'current_exchangespentamountcoins' in delta_cols:
            reward_col = 'current_exchangespentamountcoins'
        elif 'current_spend' in delta_cols:
            print(f"  📝 Using 'current_spend' as fallback for reward column")
            reward_col = 'current_spend'
        else:
            raise ValueError(f"Delta table missing reward column. Need 'current_exchangespentamountcoins' or 'current_spend'. Available: {list(delta_df.columns)}")
        
    except Exception as e:
        print(f"  ❌ Failed to read Delta table {delta_path}: {e}")
        raise e
    
    # Step 3: Find FIRST session AFTER decision_date for each user with pending decision
    print(f"\n🔗 Finding first session after each decision...")
    print(f"   Using reward column: {reward_col}")
    
    # Select reward-related columns from Delta
    # Use current_exchangespentamountcoins (or fallback) as the reward
    delta_sessions = delta_df.select(
        F.col("user_id").alias("delta_user_id"),
        F.col("session_date").alias("delta_session_date"),
        F.col(reward_col).alias("session_spend")
    )
    
    # Join decisions with Delta sessions where session_date > decision_date
    # This finds all sessions that happened AFTER the decision
    # NOTE: decision_date is STRING, delta_session_date may be DATE - cast for safe comparison
    decisions_with_sessions = decisions_df.alias("d").join(
        delta_sessions.alias("s"),
        on=(F.col("d.user_id") == F.col("s.delta_user_id")) & 
           (F.col("s.delta_session_date") > F.to_date(F.col("d.decision_date"))),
        how="inner"
    )
    
    # For each decision, find the FIRST (minimum) session_date after the decision
    # This is the reward session
    window_spec = Window.partitionBy("d.user_id", "d.decision_date").orderBy("s.delta_session_date")
    
    first_sessions = decisions_with_sessions.withColumn(
        "row_num", F.row_number().over(window_spec)
    ).filter(F.col("row_num") == 1).drop("row_num")
    
    # Create the rewards DataFrame with proper column selection
    rewards_df = first_sessions.select(
        F.col("d.user_id").alias("user_id"),
        F.col("d.decision_date").alias("decision_date"),
        F.col("d.features").alias("features"),
        F.col("d.chosen_arm").alias("chosen_arm"),
        F.col("d.action").alias("action"),
        F.col("d.arm_probabilities").alias("arm_probabilities"),
        F.col("d.model_id").alias("model_id"),
        F.col("d.timestamp").alias("timestamp"),
        F.col("s.session_spend").alias("reward"),
        F.col("s.delta_session_date").alias("reward_session_date")
    )
    
    reward_count = rewards_df.count()
    no_reward_count = decision_count - reward_count
    
    print(f"\n📊 Reward Discovery Results:")
    print(f"  ✅ Found rewards for {reward_count:,} decisions")
    print(f"  ⏳ Still pending (user hasn't played since decision): {no_reward_count:,}")
    
    if reward_count == 0:
        print(f"\n  ⚠️  No decisions ready for training (all users haven't logged in yet)")
        return spark.createDataFrame([], schema=StructType([]))
    
    # Step 4: Parse JSON columns (features, arm_probabilities) using Spark
    print(f"\n🔧 Parsing JSON columns...")
    
    # Define schema for features (map of feature_name -> value)
    feature_schema = MapType(StringType(), DoubleType())
    
    # Parse features JSON
    rewards_df = rewards_df.withColumn(
        "features_map",
        F.from_json(F.col("features"), feature_schema)
    )
    
    # Parse arm_probabilities JSON
    rewards_df = rewards_df.withColumn(
        "arm_probs_map",
        F.from_json(F.col("arm_probabilities"), feature_schema)
    )
    
    # Extract individual features as columns
    for feature in selected_features:
        rewards_df = rewards_df.withColumn(
            feature,
            F.col(f"features_map.{feature}").cast(DoubleType())
        )
    
    # Fill nulls with 0.0 for features
    feature_fill_dict = {feature: 0.0 for feature in selected_features}
    rewards_df = rewards_df.fillna(feature_fill_dict)
    
    print(f"  ✅ Extracted {len(selected_features)} features")
    
    # Step 5: Compute reward statistics
    reward_stats = rewards_df.select(
        F.mean("reward").alias("mean_reward"),
        F.stddev("reward").alias("std_reward"),
        F.min("reward").alias("min_reward"),
        F.max("reward").alias("max_reward")
    ).first()
    
    print(f"\n📊 Reward Statistics:")
    if reward_stats.mean_reward is not None:
        print(f"  Mean: {reward_stats.mean_reward:.2f} coins")
        print(f"  Std:  {(reward_stats.std_reward or 0):.2f} coins")
        print(f"  Min:  {reward_stats.min_reward:.2f} coins")
        print(f"  Max:  {reward_stats.max_reward:.2f} coins")
    else:
        print(f"  ⚠️  All rewards are NULL - statistics unavailable")
    
    print(f"\n✅ Ready for training: {reward_count:,} decisions with rewards")
    
    return rewards_df


def update_decisions_with_rewards(
    rewards_df: 'DataFrame',
    decisions_table: str
) -> int:
    """
    Update the decisions table with discovered rewards.
    
    This marks decisions as "rewarded" so they won't be processed again.
    
    Args:
        rewards_df: DataFrame with decisions that now have rewards
                   Must have columns: user_id, decision_date, reward, reward_session_date
        decisions_table: SQL table to update
        
    Returns:
        Number of decisions updated
    """
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    
    print("\n" + "=" * 80)
    print("UPDATING DECISIONS TABLE WITH REWARDS")
    print("=" * 80)
    
    spark = SparkSession.builder.appName("Phase6OnlineLearning").getOrCreate()
    
    # Deduplicate by (user_id, decision_date) to prevent MERGE failures
    # Phase 5 may have created duplicates if run multiple times
    # Keep the LATEST decision (by timestamp) for each (user_id, decision_date) pair
    from pyspark.sql.window import Window
    dedup_window = Window.partitionBy("user_id", "decision_date").orderBy(F.col("timestamp").desc())
    
    deduped_df = rewards_df.withColumn("_row_num", F.row_number().over(dedup_window)) \
        .filter(F.col("_row_num") == 1) \
        .drop("_row_num")
    
    # Create temp view for the rewards data
    deduped_df.select(
        "user_id", 
        "decision_date", 
        F.col("reward").alias("reward_amount"),
        "reward_session_date"
    ).createOrReplaceTempView("rewards_to_update")
    
    update_count = deduped_df.count()
    original_count = rewards_df.count()
    if original_count != update_count:
        print(f"  ⚠️  Deduplicated {original_count - update_count} duplicate decisions")
    
    # Use MERGE to update decisions with rewards
    try:
        merge_sql = f"""
        MERGE INTO {decisions_table} AS target
        USING rewards_to_update AS source
        ON target.user_id = source.user_id 
           AND target.decision_date = source.decision_date
        WHEN MATCHED THEN UPDATE SET
            target.reward_amount = source.reward_amount,
            target.reward_session_date = source.reward_session_date
        """
        
        spark.sql(merge_sql)
        print(f"  ✅ Updated {update_count:,} decisions with rewards")
        
    except Exception as e:
        print(f"  ⚠️  MERGE failed, trying UPDATE approach: {e}")
        
        # Fallback: collect and update row by row (slower but more compatible)
        try:
            rewards_collected = rewards_df.select(
                "user_id", "decision_date", "reward", "reward_session_date"
            ).collect()
            
            for row in rewards_collected:
                # Escape single quotes to prevent SQL injection
                safe_user_id = str(row.user_id).replace("'", "''")
                safe_decision_date = str(row.decision_date).replace("'", "''")
                safe_reward_session_date = str(row.reward_session_date).replace("'", "''")
                # Handle NULL reward
                reward_value = row.reward if row.reward is not None else 0.0
                
                update_sql = f"""
                UPDATE {decisions_table}
                SET reward_amount = {reward_value},
                    reward_session_date = '{safe_reward_session_date}'
                WHERE user_id = '{safe_user_id}' 
                  AND decision_date = '{safe_decision_date}'
                """
                spark.sql(update_sql)
            
            print(f"  ✅ Updated {update_count:,} decisions (row-by-row)")
            
        except Exception as e2:
            print(f"  ❌ Failed to update decisions: {e2}")
            raise e2
    
    return update_count


def convert_decisions_to_vw_spark(
    decisions_df: 'DataFrame',
    selected_features: List[str],
    output_path: Path,
) -> int:
    """
    Convert logged decisions to VW format using Spark RDD (no Pandas).

    Args:
        decisions_df: Spark DataFrame with decisions + rewards
        selected_features: List of feature names
        output_path: Where to write VW file

    Returns:
        Number of examples written
    """
    print("\n" + "=" * 80)
    print(f"STEP 2: CONVERTING TO VW FORMAT (SPARK RDD)")
    print("=" * 80)

    n_examples = decisions_df.count()
    print(f"\n  Converting {n_examples:,} decisions using Spark RDD...")

    def format_vw_example(row):
        """Format single row as VW CB_ADF example."""
        # Shared features
        shared_features = []
        for feature in selected_features:
            value = getattr(row, feature, 0.0)
            if value is not None and value != 0:
                clean_name = feature.replace(' ', '_').replace(':', '_').replace('|', '_')
                shared_features.append(f"{clean_name}:{float(value):.6f}")
        
        shared_line = "shared |f " + " ".join(shared_features)
        
        # Action lines
        chosen_arm = row.chosen_arm
        reward = float(row.reward)
        cost = -reward  # VW minimizes cost
        
        # Parse arm probabilities
        arm_probs_map = row.arm_probs_map
        if arm_probs_map is None:
            arm_probs_map = {}
        
        action_lines = []
        for arm_idx, arm in enumerate(ARM_ORDER):
            prob = float(arm_probs_map.get(arm, 0.2))  # Default uniform if missing
            
            if str(arm_idx) == str(chosen_arm):
                # Observed arm: has cost and probability
                # Note: chosen_arm is stored as INDEX string ("0", "1", etc.) in decisions table
                action_line = f"{arm_idx}:{cost:.4f}:{prob:.6f} |a arm:{arm_idx}"
            else:
                # Not observed: no cost
                action_line = f"{arm_idx}:0:{prob:.6f} |a arm:{arm_idx}"
            
            action_lines.append(action_line)
        
        # Return complete example
        return shared_line + "\n" + "\n".join(action_lines) + "\n\n"
    
    # Convert using RDD
    vw_examples = decisions_df.rdd.map(format_vw_example)
    
    # Write to file (coalesce to single file for efficiency)
    temp_output = str(output_path) + "_temp"
    vw_examples.coalesce(1).saveAsTextFile(temp_output)
    
    # Merge part files into single VW file
    part_files = glob.glob(f"{temp_output}/part-*")
    
    if not part_files:
        raise RuntimeError(f"No part files found in {temp_output}")
    
    with open(output_path, 'w') as outfile:
        for part_file in sorted(part_files):
            with open(part_file, 'r') as infile:
                outfile.write(infile.read())
    
    # Cleanup temp directory
    shutil.rmtree(temp_output)
    
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"\n  ✅ Wrote {n_examples:,} examples ({file_size_mb:.1f} MB)")

    return n_examples


def incremental_vw_update(
    previous_model: Path,
    new_data: Path,
    output_model: Path,
    learning_rate: float = 0.001,
) -> Tuple[bool, str]:
    """
    Incremental VW model update.

    Args:
        previous_model: Previous day's VW model
        new_data: New training data (VW format)
        output_model: Where to save updated model
        learning_rate: Learning rate for online update (lower than offline)

    Returns:
        (success, log_output)
    """
    print("\n" + "=" * 80)
    print(f"STEP 4: INCREMENTAL VW UPDATE")
    print("=" * 80)

    print(f"\n  Loading previous model: {previous_model}")
    print(f"  New data: {new_data}")
    print(f"  Output model: {output_model}")
    print(f"  Learning rate: {learning_rate}")

    # Build VW command
    cmd = f"""
    vw --cb_explore_adf \\
       --cb_type dr \\
       --initial_regressor {previous_model} \\
       --learning_rate {learning_rate} \\
       --passes 1 \\
       -d {new_data} \\
       -f {output_model} \\
       --quiet
    """

    cmd = " ".join(cmd.split())  # Clean up whitespace

    print(f"\n  Running VW update...")
    print(f"    Command: {cmd[:100]}...")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        success = result.returncode == 0

        if success:
            print(f"\n  ✅ Model update successful!")
        else:
            print(f"\n  ❌ Model update failed!")
            print(f"     Error: {result.stderr[:200]}")

        return success, result.stderr

    except subprocess.TimeoutExpired:
        print(f"\n  ❌ Model update timeout (5 minutes)")
        return False, "Timeout"
    except Exception as e:
        print(f"\n  ❌ Model update error: {e}")
        return False, str(e)


def validate_updated_model(
    updated_model: Path,
    validation_vw: Path,
) -> Dict[str, float]:
    """
    Validate updated model on validation set.

    Args:
        updated_model: Newly updated VW model
        validation_vw: Validation VW file

    Returns:
        Dict with validation metrics
    """
    print("\n" + "=" * 80)
    print(f"STEP 5: VALIDATING UPDATED MODEL")
    print("=" * 80)

    # Compute progressive validation loss
    cmd = f"vw -i {updated_model} -t -d {validation_vw} --quiet"

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60
        )

        # Parse loss from output
        loss = None
        for line in result.stderr.split('\n'):
            if 'average loss' in line.lower():
                try:
                    loss = float(line.split('=')[-1].strip())
                    break
                except:
                    pass

        if loss is None:
            loss = float('inf')

        print(f"\n  📊 Validation Metrics:")
        print(f"     Average loss: {loss:.4f}")

        return {'validation_loss': loss}

    except Exception as e:
        print(f"\n  ⚠️  Validation error: {e}")
        return {'validation_loss': float('inf')}


def should_deploy_model(
    updated_model: Path,
    validation_metrics: Dict[str, float],
    threshold_pct: float = 5.0
) -> Tuple[bool, str]:
    """
    Determine if model should be deployed based on validation.
    
    Compares validation loss to previous run. Skips deployment if loss
    increased by more than threshold_pct.
    
    Args:
        updated_model: New model to potentially deploy
        validation_metrics: Validation metrics for new model
        threshold_pct: Maximum allowed loss increase percentage (default: 5%)
        
    Returns:
        (should_deploy, reason)
    """
    # Load previous validation loss from history
    history_file = LOGS_DIR / "online_learning_history.jsonl"
    previous_loss = None
    
    if history_file.exists():
        try:
            with open(history_file, 'r') as f:
                lines = f.readlines()
                if lines:
                    # Get last successful deployment
                    for line in reversed(lines):
                        entry = json.loads(line)
                        if entry.get('deployed') and entry.get('validation_loss'):
                            previous_loss = float(entry['validation_loss'])
                            break
        except Exception as e:
            print(f"  ⚠️  Could not load previous validation loss: {e}")
    
    current_loss = validation_metrics['validation_loss']
    
    # First deployment (no history)
    if previous_loss is None or previous_loss == float('inf'):
        return True, "First deployment (no history)"
    
    # Check if loss increased
    loss_increase_pct = ((current_loss - previous_loss) / previous_loss) * 100
    
    if loss_increase_pct > threshold_pct:
        return False, f"Loss increased by {loss_increase_pct:.2f}% (threshold: {threshold_pct}%)"
    
    return True, f"Loss change: {loss_increase_pct:+.2f}%"


def deploy_model_with_rollback(
    updated_model: Path,
    validation_metrics: Dict[str, float],
    dry_run: bool = False,
    threshold_pct: float = 5.0
) -> Tuple[bool, str]:
    """
    Deploy model with validation check and rollback.
    
    Args:
        updated_model: New model to deploy
        validation_metrics: Validation metrics for new model
        dry_run: If True, don't actually deploy
        threshold_pct: Maximum allowed loss increase percentage
        
    Returns:
        (deployed, reason)
    """
    print("\n" + "=" * 80)
    print(f"STEP 3: VALIDATING \u0026 DEPLOYING MODEL")
    print("=" * 80)
    
    # Check if should deploy
    should_deploy, reason = should_deploy_model(updated_model, validation_metrics, threshold_pct)
    
    if not should_deploy:
        print(f"\n⚠️  DEPLOYMENT SKIPPED: {reason}")
        print(f"   Keeping previous model in production")
        
        # Log alert
        alert = {
            'timestamp': datetime.utcnow().isoformat(),
            'severity': 'WARNING',
            'message': f'Model deployment skipped: {reason}',
            'validation_loss': validation_metrics['validation_loss'],
            'model_path': str(updated_model)
        }
        
        alert_file = LOGS_DIR / "deployment_alerts.jsonl"
        with open(alert_file, 'a') as f:
            f.write(json.dumps(alert) + '\n')
        
        print(f"   ✅ Alert logged to: {alert_file}")
        
        return False, reason
    
    # Proceed with deployment
    print(f"\n✅ Validation passed: {reason}")
    
    production_model = get_latest_model(MODELS_DIR)
    backup_model = MODELS_DIR / f"vw_bandit_dr_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.vw"
    
    if dry_run:
        print(f"\n🔶 DRY RUN - Model not deployed")
        print(f"   Would deploy: {updated_model} → {production_model}")
        return True, "Dry run"
    
    try:
        # Create backup
        if production_model.exists():
            shutil.copy(production_model, backup_model)
            print(f"\n✅ Backup created: {backup_model}")
        
        # Atomic swap
        shutil.copy(updated_model, production_model)
        
        # Verify
        if not production_model.exists() or production_model.stat().st_size == 0:
            raise Exception("Deployment verification failed")
        
        print(f"\n✅ Model deployed: {production_model}")
        
        # Cleanup old backups (keep last 10 days)
        cleanup_old_backups(MODELS_DIR, days_to_keep=10)
        
        return True, reason
        
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        
        # Rollback
        if backup_model.exists():
            shutil.copy(backup_model, production_model)
            print(f"✅ Rolled back to: {backup_model}")
        
        return False, f"Deployment failed: {e}"


def cleanup_old_backups(models_dir: Path, days_to_keep: int = 10):
    """
    Cleanup old backup models, keeping only the last N days.
    
    Args:
        models_dir: Directory containing model backups
        days_to_keep: Number of days of backups to keep
    """
    try:
        backup_pattern = models_dir / "vw_bandit_dr_backup_*.vw"
        backup_files = list(models_dir.glob("vw_bandit_dr_backup_*.vw"))
        
        if not backup_files:
            return
        
        # Sort by modification time (newest first)
        backup_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Keep last N days (assuming 1 backup per day)
        files_to_keep = backup_files[:days_to_keep]
        files_to_delete = backup_files[days_to_keep:]
        
        if files_to_delete:
            print(f"\n🗑️  Cleaning up old backups (keeping last {days_to_keep} days)...")
            for backup_file in files_to_delete:
                backup_file.unlink()
                print(f"   Deleted: {backup_file.name}")
            print(f"   ✅ Cleaned up {len(files_to_delete)} old backups")
    
    except Exception as e:
        print(f"  ⚠️  Backup cleanup failed: {e}")


def deploy_model(
    updated_model: Path,
    dry_run: bool = False,
) -> bool:
    """
    DEPRECATED: Use deploy_model_with_rollback instead.
    
    Deploy updated model with atomic swap.

    Args:
        updated_model: New model to deploy
        dry_run: If True, don't actually deploy

    Returns:
        Success status
    """
    print("\n" + "=" * 80)
    print(f"STEP 6: DEPLOYING MODEL")
    print("=" * 80)

    production_model = get_latest_model(MODELS_DIR)
    backup_model = MODELS_DIR / f"vw_bandit_dr_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.vw"

    if dry_run:
        print(f"\n  🔶 DRY RUN - Model not deployed")
        print(f"     Would deploy: {updated_model} → {production_model}")
        return True

    print(f"\n  Creating backup: {backup_model}")
    if production_model.exists():
        shutil.copy(production_model, backup_model)
        print(f"    ✅ Backup created")

    print(f"\n  Deploying new model (atomic swap)...")
    shutil.copy(updated_model, production_model)
    print(f"    ✅ Model deployed: {production_model}")

    # Verify deployment
    if production_model.exists() and production_model.stat().st_size > 0:
        print(f"\n  ✅ Deployment successful!")
        return True
    else:
        print(f"\n  ❌ Deployment failed - rolling back...")
        if backup_model.exists():
            shutil.copy(backup_model, production_model)
            print(f"    ✅ Rolled back to backup")
        return False


def log_online_learning_metrics(
    target_date: datetime,
    n_examples: int,
    validation_metrics: Dict[str, float],
    deployed: bool,
):
    """Log online learning metrics for monitoring."""
    log_entry = {
        'date': target_date.strftime('%Y-%m-%d'),
        'timestamp': datetime.utcnow().isoformat(),
        'n_examples': n_examples,
        'validation_loss': validation_metrics.get('validation_loss'),
        'deployed': deployed,
    }

    log_file = LOGS_DIR / "online_learning_history.jsonl"
    with open(log_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

    print(f"\n  ✅ Logged metrics: {log_file}")


def main():
    """Run daily online learning cycle."""

    args = parse_args()

    print("\n" + "=" * 80)
    print(" " * 20 + "DAILY ONLINE LEARNING CYCLE")
    print(" " * 30 + "PHASE 6")
    print("=" * 80)

    # Get target date
    target_date = get_target_date(args.date)
    print(f"\nTarget date: {target_date.strftime('%Y-%m-%d')}")
    if args.dry_run:
        print("🔶 DRY RUN MODE - Model will not be deployed")

    # Load selected features from argument
    print(f"\n📥 Loading selected features from: {args.selected_features}")
    
    # Normalize path
    if args.selected_features.startswith("dbfs:/"):
        features_path = "/dbfs" + args.selected_features[5:]
    else:
        features_path = args.selected_features
    
    try:
        with open(features_path, 'r') as f:
            selected_features = json.load(f)['selected_features']
        selected_features = [f for f in selected_features if f != "action"]
        print(f"✅ Loaded {len(selected_features)} features")
    except Exception as e:
        print(f"❌ Failed to load selected features: {e}")
        sys.exit(1)

    # Step 1: Aggregate decisions and fetch rewards (Spark-native)
    try:
        decisions_with_rewards = aggregate_decisions_and_rewards(
            target_date=target_date,
            decisions_table="spaceplay.bandit_decisions",
            delta_path=args.delta_path,
            selected_features=selected_features
        )
    except Exception as e:
        print(f"\n❌ Failed to aggregate decisions/rewards: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # IMPORTANT: Cache the DataFrame to prevent re-evaluation issues
    # After update_decisions_with_rewards() runs MERGE, the DataFrame can become stale
    # Caching ensures the DataFrame is materialized in memory and won't be re-evaluated
    decisions_with_rewards.cache()

    # Check if we have any decisions ready for training
    n_decisions = decisions_with_rewards.count() if hasattr(decisions_with_rewards, 'count') else 0

    if n_decisions == 0:
        decisions_with_rewards.unpersist()  # Clean up cache
        print("\n" + "=" * 80)
        print("⏳ NO DECISIONS READY FOR TRAINING")
        print("=" * 80)
        print(f"\n  No users have logged in since their decisions were made.")
        print(f"  This is expected behavior - model will train when rewards are available.")
        
        # Still trigger Phase 5 with old model if requested
        if args.trigger_phase5:
            print(f"\n  Triggering Phase 5 with CURRENT model (no update)...")
            trigger_phase5(args, target_date, get_latest_model(MODELS_DIR))
        
        print("=" * 80 + "\n")
        return  # Exit gracefully - no training needed

    # CHECK IF TRAINING IS SKIPPED
    # IMPORTANT: In skip-training mode, update decisions table immediately
    # since we won't reach the post-training update
    if args.skip_training:
        print("\n" + "=" * 80)
        print("⏭️  SKIPPING MODEL TRAINING (--skip-training flag set)")
        print("=" * 80)

        # Update decisions table in skip-training mode
        n_updated = 0
        update_failed = False
        try:
            print(f"\n📝 Updating decisions table with {n_decisions:,} rewards...")
            n_updated = update_decisions_with_rewards(
                decisions_with_rewards,
                "spaceplay.bandit_decisions"
            )
            print(f"  ✅ Updated {n_updated:,} decisions with rewards")
        except Exception as e:
            update_failed = True
            print(f"\n  ⚠️  Decisions table update FAILED: {e}")
            import traceback
            traceback.print_exc()

        if update_failed:
            print(f"\n  ⚠️  {n_decisions:,} rewards NOT written to decisions table.")
        else:
            print(f"\n  ✅ Decisions table updated: {n_updated:,} of {n_decisions:,} rewards written.")

        print(f"  Model will NOT be updated until --skip-training is removed.")
        print(f"  Use this mode until the model is being served in production.")

        # Trigger Phase 5 with current model if requested
        if args.trigger_phase5:
            print(f"\n  Triggering Phase 5 with CURRENT model (no update)...")
            trigger_phase5(args, target_date, get_latest_model(MODELS_DIR))

        # Clean up cached DataFrame
        decisions_with_rewards.unpersist()

        print("\n" + "=" * 80)
        print("PHASE 6 COMPLETE (training skipped)")
        print("=" * 80 + "\n")
        return

    # Step 2: Convert to VW format (Spark RDD)
    # NOTE: Decisions table is NOT updated yet - will update after training succeeds
    online_vw_path = DATA_PROCESSED / f"online_{target_date.strftime('%Y%m%d')}.vw"
    try:
        n_examples = convert_decisions_to_vw_spark(
            decisions_with_rewards,
            selected_features,
            online_vw_path
        )
    except Exception as e:
        print(f"\n❌ VW conversion failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Step 3: Incremental update
    print("\n" + "=" * 80)
    print(f"STEP 3: INCREMENTAL VW UPDATE")
    print("=" * 80)
    
    if args.model_path:
        previous_model = Path(args.model_path)
        print(f"\n  Using specific base model from args: {previous_model}")
    else:
        previous_model = get_latest_model(MODELS_DIR)
        print(f"\n  Using latest detected base model: {previous_model}")

    updated_model = MODELS_DIR / f"vw_bandit_dr_{target_date.strftime('%Y%m%d')}.vw"

    success, log_output = incremental_vw_update(
        previous_model, online_vw_path, updated_model, learning_rate=0.001
    )

    if not success:
        print(f"\n❌ Online learning failed!")
        print(f"   See logs for details")

        # Still trigger Phase 5 with old model if requested
        if args.trigger_phase5:
            print(f"\n  Triggering Phase 5 with CURRENT model (training failed)...")
            trigger_phase5(args, target_date, previous_model)

        # Clean up cached DataFrame before exit
        decisions_with_rewards.unpersist()

        sys.exit(1)

    # Step 4: Validate updated model
    print("\n" + "=" * 80)
    print(f"STEP 4: VALIDATING UPDATED MODEL")
    print("=" * 80)
    
    validation_vw = DATA_PROCESSED / "valid.vw"
    validation_metrics = validate_updated_model(updated_model, validation_vw)

    # Step 5: Deploy model with validation check and rollback
    deployed, deploy_reason = deploy_model_with_rollback(
        updated_model,
        validation_metrics,
        dry_run=args.dry_run,
        threshold_pct=5.0
    )

    # Log metrics
    log_online_learning_metrics(target_date, n_examples, validation_metrics, deployed)

    # Step 6: Update decisions table ONLY IF training succeeded
    # CRITICAL: This ensures we don't lose training data if any step fails
    # Decisions stay with reward_amount IS NULL and will be retried next run
    print("\n" + "=" * 80)
    print(f"STEP 6: UPDATING DECISIONS TABLE")
    print("=" * 80)

    n_updated = 0
    update_success = False

    if deployed and not args.dry_run:
        try:
            print(f"\n📝 Updating {n_decisions:,} decisions with rewards...")
            n_updated = update_decisions_with_rewards(
                decisions_with_rewards,
                "spaceplay.bandit_decisions"
            )
            update_success = True
            print(f"  ✅ Successfully updated {n_updated:,} decisions with rewards")
        except Exception as e:
            print(f"\n  ⚠️  WARNING: Failed to update decisions table: {e}")
            print(f"      Training succeeded but decisions not marked as rewarded")
            print(f"      These {n_decisions:,} decisions will be retrained next run")
            # Don't fail the entire process - training succeeded
    else:
        if args.dry_run and deployed:
            print(f"\n🔶 Skipping decisions table update - DRY RUN mode")
            print(f"    In production, would update {n_decisions:,} decisions")
            print(f"    These {n_decisions:,} decisions will be retried next run")
        else:
            print(f"\n⏭️  Skipping decisions table update - model not deployed")
            print(f"    Reason: {deploy_reason}")
            print(f"    These {n_decisions:,} decisions will be retried next run")

    # Step 7: Trigger Phase 5 (with updated or old model depending on deployment)
    if args.trigger_phase5:
        model_to_use = updated_model if deployed else previous_model
        model_status = "UPDATED" if deployed else "CURRENT"
        
        print(f"\n🚀 Triggering Phase 5 with {model_status} model...")
        trigger_phase5(args, target_date, model_to_use)

    # Summary
    print("\n" + "=" * 80)
    print("ONLINE LEARNING COMPLETE!")
    print("=" * 80)
    print(f"\n✅ Updated model: {updated_model}")
    print(f"✅ Deployed: {deployed} ({deploy_reason})")
    print(f"✅ Examples processed: {n_examples:,}")
    print(f"✅ Validation loss: {validation_metrics['validation_loss']:.4f}")

    if deployed and update_success:
        print(f"✅ Decisions updated: {n_updated:,}")
    elif deployed and not update_success:
        print(f"⚠️  Decisions NOT updated (update failed, will retry)")
    else:
        print(f"⏭️  Decisions NOT updated (model not deployed)")

    if not args.dry_run and deployed:
        print(f"\n🎯 Model is now live in production!")
        print(f"   Phase 5 will use the updated model for predictions")
        if update_success:
            print(f"   {n_updated:,} decisions marked as rewarded and trained")
    elif not deployed:
        print(f"\n⚠️  Model not deployed - Phase 5 will use previous model")
        print(f"   {n_decisions:,} decisions will be retried next run")

    # Clean up cached DataFrame
    decisions_with_rewards.unpersist()

    print("=" * 80 + "\n")


def trigger_phase5(args, target_date: datetime, model_path: Path):
    """
    Trigger Phase 5 batch inference job.
    
    Args:
        args: Command line arguments
        target_date: Date for inference
        model_path: Path to VW model to use
    """
    try:
        import requests
        
        # Get Databricks credentials
        databricks_token = os.environ.get("DATABRICKS_TOKEN")
        if not databricks_token:
            print("  ⚠️  DATABRICKS_TOKEN not set - cannot trigger Phase 5")
            return
        
        workspace_url = os.environ.get("DATABRICKS_HOST", "https://adb-249008710733422.2.azuredatabricks.net")
        if not workspace_url.startswith("http"):
            workspace_url = f"https://{workspace_url}"
        
        # Trigger Phase 5 job (Cloudflare KV version)
        api_url = f"{workspace_url}/api/2.1/jobs/run-now"
        payload = {
            "job_id": args.phase5_job_id,
            "python_params": [
                "--delta-path", args.delta_path,
                "--date", target_date.strftime('%Y-%m-%d'),
                "--selected-features", args.selected_features,
                # NOTE: No --model-path - Phase 5 will auto-detect latest model
                # NOTE: No --kv-config - Phase 5 will auto-detect from default path (dbfs:/mnt/bandit/config/cloudflare_kv_config.json)
                "--table-name", "spaceplay.user_multipliers",
                "--decisions-table", "spaceplay.bandit_decisions",
                "--inference"
            ]
        }
        
        response = requests.post(
            api_url,
            headers={"Authorization": f"Bearer {databricks_token}"},
            json=payload,
            timeout=10
        )
        response.raise_for_status()
        run_id = response.json().get("run_id")
        
        print(f"  ✅ Triggered Phase 5 (Job ID: {args.phase5_job_id}, Run ID: {run_id})")
        print(f"  🔍 Phase 5 will auto-detect latest model: {model_path.name}")
        print(f"  📅 Date: {target_date.strftime('%Y-%m-%d')}")
        
    except Exception as e:
        print(f"  ⚠️  Failed to trigger Phase 5: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()