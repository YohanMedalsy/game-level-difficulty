#!/usr/bin/env python3
"""
Phase 5: Batch Prediction (Direct VW Loading)

Runs batch predictions using VW model loaded directly in-process.
No FastAPI overhead - optimized for speed.

Usage:
  # Training/Testing (can rerun same date)
  python scripts/05_run_fastapi_batch.py \
    --delta-path dbfs:/mnt/features/daily_features.delta \
    --date 2025-10-20 \
    --selected-features dbfs:/mnt/artifacts/selected_features_50.json \
    --model-path dbfs:/mnt/vw_pipeline/models_aug15/vw_bandit_dr_best.vw
  
  # Inference mode (idempotency enabled)
  python scripts/05_run_fastapi_batch.py \
    --delta-path dbfs:/mnt/features/daily_features.delta \
    --date 2025-10-20 \
    --selected-features dbfs:/mnt/artifacts/selected_features_50.json \
    --model-path dbfs:/mnt/vw_pipeline/models_aug15/vw_bandit_dr_best.vw \
    --inference
"""

import os
import sys
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np

# Add parent to path
CURRENT_FILE = Path(__file__).resolve()
BANDIT_ROOT = CURRENT_FILE.parent.parent.absolute()
sys.path.insert(0, str(BANDIT_ROOT))

from src.constants import (
    ACTION_TO_DELTA,
    ARM_ORDER,
    get_valid_arms,
    compute_next_multiplier,
    assign_feature_to_namespace,
)

# Try to import vowpalwabbit
try:
    import vowpalwabbit as vw
    VW_AVAILABLE = True
except ImportError:
    VW_AVAILABLE = False
    print("⚠️  vowpalwabbit not installed. Install with: pip install vowpalwabbit==9.10.0")


def _to_local_path(path: str) -> Path:
    """Convert DBFS path to local mount path."""
    if path.startswith("dbfs:/"):
        return Path("/dbfs" + path[5:])
    return Path(path)


def maybe_skip_by_date(state_path: str, target_date: str) -> bool:
    """Return True if target_date <= last processed date stored at state_path."""
    if not state_path:
        return False
    try:
        p = _to_local_path(state_path)
        if p.exists():
            txt = p.read_text().strip()
            if txt:
                last_dt = datetime.strptime(txt, "%Y-%m-%d").date()
                curr_dt = datetime.strptime(target_date, "%Y-%m-%d").date()
                if curr_dt <= last_dt:
                    print(f"🚫 Date {target_date} already processed (last={txt}); skipping run.")
                    return True
    except Exception as exc:
        print(f"⚠️  Could not read state from {state_path}: {exc}")
    return False


def update_last_processed(state_path: str, target_date: str) -> None:
    if not state_path:
        return
    try:
        p = _to_local_path(state_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(target_date)
        print(f"📝 Updated last processed date to {target_date} at {state_path}")
    except Exception as exc:
        print(f"⚠️  Failed to update state at {state_path}: {exc}")


def load_selected_features(features_json_path: str = None) -> List[str]:
    """Load selected features list."""
    if features_json_path:
        features_path = _to_local_path(features_json_path)
    else:
        features_path = BANDIT_ROOT / "artifacts" / "selected_features_50.json"
    
    if not features_path.exists():
        raise FileNotFoundError(f"Selected features not found: {features_path}")
    
    with open(features_path, 'r') as f:
        data = json.load(f)
    
    features = data['selected_features']
    print(f"✅ Loaded {len(features)} selected features from {features_path}")
    return features


def load_users_from_delta(delta_path: str, target_date: str, selected_features: List[str]) -> List[Dict]:
    """Load users from Delta table for a specific date."""
    from pyspark.sql import SparkSession
    
    spark = SparkSession.builder.appName("Phase5LoadUsers").getOrCreate()
    
    # Convert to local path if needed
    if delta_path.startswith("dbfs:/"):
        local_path = "/dbfs" + delta_path[5:]
    else:
        local_path = delta_path
    
    print(f"   Reading Delta table: {local_path}")
    print(f"   Partition pruning: session_date = '{target_date}'")
    
    # Read with partition pruning
    df = spark.read.format("delta").load(local_path)
    df = df.filter(f"session_date = '{target_date}'")
    
    # Check if data exists
    row_count = df.count()
    if row_count == 0:
        raise ValueError(f"No data found for date {target_date} in Delta table {delta_path}")
    
    print(f"   ✅ Found {row_count:,} rows for {target_date} (partition pruning active)")
    
    # Get unique users (one row per user for that date)
    df = df.dropDuplicates(["user_id"])
    
    # Collect to Python
    users = []
    for row in df.collect():
        user = {
            "user_id": row["user_id"],
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


def load_vw_model(model_path: str):
    """Load VW model from file."""
    if not VW_AVAILABLE:
        raise RuntimeError("vowpalwabbit not installed. Install with: pip install vowpalwabbit==9.10.0")
    
    local_path = _to_local_path(model_path)
    
    if not local_path.exists():
        raise FileNotFoundError(f"VW model not found: {local_path}")
    
    # Load model with prediction mode
    model = vw.Workspace(f"-i {str(local_path)} --quiet", enable_logging=False)
    print(f"✅ Loaded VW model from {local_path}")
    
    return model


def format_vw_example(user: Dict, selected_features: List[str], namespace_features: Dict[str, List[str]]) -> str:
    """
    Format user context as VW CB_ADF example.
    
    Returns:
        VW formatted string for prediction
    """
    current_mult = user["current_effectivelevelmultiplier"]
    valid_arms = get_valid_arms(current_mult)
    
    # Build shared context
    shared_parts = []
    for namespace, features in sorted(namespace_features.items()):
        ns_parts = []
        for feature in features:
            value = user["features"].get(feature, 0.0)
            if value != 0:  # Skip zero values for efficiency
                clean_name = feature.replace(' ', '_').replace(':', '_').replace('|', '_')
                ns_parts.append(f"{clean_name}:{value:.6f}")
        
        if ns_parts:
            shared_parts.append(f"|{namespace} " + " ".join(ns_parts))
    
    shared_line = "shared " + " ".join(shared_parts)
    
    # Build action lines
    action_lines = []
    for arm_idx, arm in enumerate(ARM_ORDER):
        delta = ACTION_TO_DELTA[arm]
        feasible = 1 if arm in valid_arms else 0
        action_features = f"|a arm:{arm_idx} delta:{delta:.2f} |c mult:{current_mult:.2f} feasible:{feasible}"
        action_line = f"{arm_idx} {action_features}"
        action_lines.append(action_line)
    
    # Combine
    vw_example = shared_line + "\n" + "\n".join(action_lines)
    return vw_example


def predict_with_vw(vw_model, vw_example: str, feasible_arms: List[str]) -> tuple:
    """
    Make prediction using VW model.
    
    Returns:
        (chosen_arm, arm_probabilities)
    """
    # Parse the multiline example and predict
    predictions = vw_model.predict(vw_example)
    
    # VW CB_ADF returns action probabilities
    # predictions is typically an array of probabilities for each action
    if isinstance(predictions, (list, np.ndarray)):
        probs_array = np.array(predictions)
        
        # Zero out infeasible arms
        for i, arm in enumerate(ARM_ORDER):
            if arm not in feasible_arms:
                probs_array[i] = 0.0
        
        # Renormalize
        total = probs_array.sum()
        if total > 0:
            probs_array = probs_array / total
        else:
            # Fallback to uniform over feasible
            probs_array = np.zeros(len(ARM_ORDER))
            for i, arm in enumerate(ARM_ORDER):
                if arm in feasible_arms:
                    probs_array[i] = 1.0 / len(feasible_arms)
        
        # Create probability dict
        arm_probs = {arm: float(probs_array[i]) for i, arm in enumerate(ARM_ORDER)}
        
        # Sample action
        chosen_idx = int(np.random.choice(np.arange(len(ARM_ORDER)), p=probs_array))
        chosen_arm = ARM_ORDER[chosen_idx]
        
    else:
        # Fallback: uniform random
        print(f"  ⚠️  Unexpected VW prediction format: {type(predictions)}, using uniform")
        uniform_prob = 1.0 / len(feasible_arms)
        arm_probs = {arm: (uniform_prob if arm in feasible_arms else 0.0) for arm in ARM_ORDER}
        chosen_arm = np.random.choice(feasible_arms)
    
    return chosen_arm, arm_probs


def sync_to_redis(results: List[Dict], redis_host: str, redis_port: int, redis_key: str):
    """
    Sync multiplier updates to Azure Redis Cache.
    """
    if not redis_host:
        print("⚠️  Redis host not provided; skipping Redis sync")
        return

    print(f"🔄 Syncing {len(results)} records to Redis ({redis_host}:{redis_port})...")
    try:
        import redis
        # Connect to Redis (SSL enabled by default for Azure)
        r = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=redis_key,
            ssl=True,
            socket_timeout=5
        )
        
        # Use pipeline for efficient batch write (single network round-trip)
        pipe = r.pipeline()
        for row in results:
            user_id = row['user_id']
            multiplier = row['current_multiplier']
            # Key: "bongo:{user_id}" → Value: multiplier
            pipe.set(f"bongo:{user_id}", str(multiplier))
        
        # Execute all commands in single batch
        pipe.execute()
        print(f"✅ Successfully synced {len(results)} records to Redis")
        
    except ImportError:
        print("❌ 'redis' package not installed. Please run: pip install redis")
    except Exception as e:
        print(f"❌ Redis sync failed: {e}")


def make_predictions(
    users: List[Dict], 
    vw_model, 
    selected_features: List[str],
    namespace_features: Dict[str, List[str]],
    table_name: str = "spaceplay.user_multipliers", 
    decisions_table: str = "spaceplay.bandit_decisions",
    redis_config: Dict = None
):
    """
    Make predictions for users and write to SQL tables + Redis.
    """
    from pyspark.sql import SparkSession
    from pyspark.sql.types import StructType, StructField, StringType, DoubleType, TimestampType
    
    spark = SparkSession.builder.appName("Phase5BatchPredictions").getOrCreate()
    
    print(f"📊 Processing {len(users)} users...")
    
    results = []
    decisions = []
    
    start_time = time.time()
    
    for i, user in enumerate(users):
        try:
            # Get current multiplier and feasible arms
            prev_multiplier = user["current_effectivelevelmultiplier"]
            feasible_arms = get_valid_arms(prev_multiplier)
            
            # Format VW example
            vw_example = format_vw_example(user, selected_features, namespace_features)
            
            # Make prediction
            chosen_arm, arm_probs = predict_with_vw(vw_model, vw_example, feasible_arms)
            
            # Calculate new multiplier
            action = ACTION_TO_DELTA[chosen_arm]
            current_multiplier = round(prev_multiplier + action, 2)
            
            # 1. Result for User Multipliers Table
            results.append({
                "user_id": str(user["user_id"]),
                "current_multiplier": float(current_multiplier),
                "prev_multiplier": float(prev_multiplier),
                "last_update_timestamp": datetime.now()
            })
            
            # 2. Decision for Phase 6 Logging
            decisions.append({
                "user_id": str(user["user_id"]),
                "decision_date": datetime.now().strftime("%Y-%m-%d"),
                "features": json.dumps(user["features"]),
                "chosen_arm": str(ARM_ORDER.index(chosen_arm)),
                "action": action,
                "arm_probabilities": json.dumps(arm_probs),
                "model_id": "vw_bandit_dr_best",
                "timestamp": datetime.now()
            })
            
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                print(f"  Processed {i + 1}/{len(users)} users ({rate:.1f} users/sec)...")
                
        except Exception as e:
            print(f"  ⚠️  Error for user {user.get('user_id', 'unknown')}: {e}")
    
    total_time = time.time() - start_time
    print(f"  ✅ Processed {len(results)} users in {total_time:.2f}s ({len(results)/total_time:.1f} users/sec)")
    
    if not results:
        print("⚠️  No results to save.")
        return []

    # --- A. Upsert User Multipliers (SQL) ---
    schema = StructType([
        StructField("user_id", StringType(), False),
        StructField("current_multiplier", DoubleType(), True),
        StructField("prev_multiplier", DoubleType(), True),
        StructField("last_update_timestamp", TimestampType(), True)
    ])
    
    df_results = spark.createDataFrame(results, schema)
    
    # Ensure database exists
    db_name = table_name.split(".")[0] if "." in table_name else "default"
    spark.sql(f"CREATE DATABASE IF NOT EXISTS {db_name}")
    
    print(f"💾 Ensuring table {table_name} exists...")
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            user_id STRING,
            current_multiplier DOUBLE,
            prev_multiplier DOUBLE,
            last_update_timestamp TIMESTAMP
        ) USING DELTA
    """)
    
    print(f"🔄 Merging {len(results)} rows into {table_name}...")
    df_results.createOrReplaceTempView("updates")
    
    spark.sql(f"""
        MERGE INTO {table_name} AS target
        USING updates AS source
        ON target.user_id = source.user_id
        WHEN MATCHED THEN
            UPDATE SET
                target.current_multiplier = source.current_multiplier,
                target.prev_multiplier = source.prev_multiplier,
                target.last_update_timestamp = source.last_update_timestamp
        WHEN NOT MATCHED THEN
            INSERT (user_id, current_multiplier, prev_multiplier, last_update_timestamp)
            VALUES (source.user_id, source.current_multiplier, source.prev_multiplier, source.last_update_timestamp)
    """)
    print(f"✅ Successfully upserted {len(results)} records to {table_name}")

    # --- B. Log Decisions (SQL) ---
    print(f"📝 Logging {len(decisions)} decisions to {decisions_table}...")
    decision_schema = StructType([
        StructField("user_id", StringType(), False),
        StructField("decision_date", StringType(), True),
        StructField("features", StringType(), True),
        StructField("chosen_arm", StringType(), True),
        StructField("action", DoubleType(), True),
        StructField("arm_probabilities", StringType(), True),
        StructField("model_id", StringType(), True),
        StructField("timestamp", TimestampType(), True)
    ])
    
    df_decisions = spark.createDataFrame(decisions, decision_schema)
    
    spark.sql(f"""
        CREATE TABLE IF NOT EXISTS {decisions_table} (
            user_id STRING,
            decision_date STRING,
            features STRING,
            chosen_arm STRING,
            action DOUBLE,
            arm_probabilities STRING,
            model_id STRING,
            timestamp TIMESTAMP
        ) USING DELTA
        PARTITIONED BY (decision_date)
    """)
    
    # Append mode for decisions log
    df_decisions.write.format("delta").mode("append").saveAsTable(decisions_table)
    print(f"✅ Successfully logged decisions to {decisions_table}")

    # --- C. Sync to Redis ---
    if redis_config and redis_config.get("host"):
        sync_to_redis(
            results, 
            redis_config["host"], 
            redis_config["port"], 
            redis_config["key"]
        )

    return results


def main():
    parser = argparse.ArgumentParser(description="Batch predictions via direct VW loading")
    
    # Required args
    parser.add_argument("--delta-path", type=str, required=True, help="Delta file from Phase 0")
    parser.add_argument("--date", type=str, required=True, help="Target date (YYYY-MM-DD)")
    parser.add_argument("--selected-features", type=str, required=True, help="Path to selected features JSON")
    parser.add_argument("--model-path", type=str, required=True, help="Path to VW model file")
    
    # Optional args
    parser.add_argument("--table-name", type=str, default="spaceplay.user_multipliers", help="Databricks SQL table to upsert results to")
    parser.add_argument("--decisions-table", type=str, default="spaceplay.bandit_decisions", help="Databricks SQL table to log decisions to")
    parser.add_argument("--trigger-phase6", action='store_true', help="Trigger Phase 6 job after predictions complete (requires --phase6-job-id)")
    parser.add_argument("--phase6-job-id", type=int, help="Databricks Job ID for Phase 6 (required if --trigger-phase6)")
    parser.add_argument("--state-path", type=str, default="dbfs:/mnt/bandit/phase5_state/last_processed_date.txt", help="State file to track last processed date")
    parser.add_argument("--inference", action='store_true', help="Inference mode: enables idempotency checks")
    
    # Redis args
    parser.add_argument("--redis-host", type=str, help="Azure Redis Hostname")
    parser.add_argument("--redis-port", type=int, default=6380, help="Azure Redis Port")
    parser.add_argument("--redis-key", type=str, help="Azure Redis Access Key")
    
    args = parser.parse_args()
    
    # Redis config from args or env
    redis_config = {
        "host": args.redis_host or os.environ.get("REDIS_HOST"),
        "port": args.redis_port or int(os.environ.get("REDIS_PORT", 6380)),
        "key": args.redis_key or os.environ.get("REDIS_KEY")
    }
    
    print("=" * 80)
    print("PHASE 5: BATCH PREDICTIONS (DIRECT VW)")
    print("=" * 80)
    print(f"\n📅 Target date: {args.date}")
    print(f"📂 Delta file: {args.delta_path}")
    print(f"🤖 Model: {args.model_path}")
    
    # Skip if date already processed (ONLY in inference mode)
    if args.inference and maybe_skip_by_date(args.state_path, args.date):
        print("   (Inference mode: idempotency enabled)")
        return 0
    
    if not args.inference:
        print("   (Training/Testing mode: idempotency disabled)")
    
    try:
        # Load selected features
        print(f"\n📥 Loading selected features...")
        selected_features = load_selected_features(args.selected_features)
        
        # Organize into namespaces
        from collections import defaultdict
        ns_dict = defaultdict(list)
        for feature in selected_features:
            ns = assign_feature_to_namespace(feature)
            ns_dict[ns].append(feature)
        namespace_features = dict(ns_dict)
        print(f"   ✅ Organized into {len(namespace_features)} namespaces")
        
        # Load VW model
        print(f"\n🤖 Loading VW model...")
        vw_model = load_vw_model(args.model_path)
        
        # Load users
        print(f"\n📥 Loading users from Delta file...")
        users = load_users_from_delta(args.delta_path, args.date, selected_features)
        print(f"   Found {len(users)} users")
        
        # Make predictions and write to SQL + Redis
        results = make_predictions(
            users, 
            vw_model,
            selected_features,
            namespace_features,
            args.table_name, 
            args.decisions_table,
            redis_config
        )
        
        # Update state ONLY in inference mode
        if args.inference:
            update_last_processed(args.state_path, args.date)
        
        # Trigger Phase 6 if requested
        if args.trigger_phase6:
            if not args.phase6_job_id:
                print("⚠️  --trigger-phase6 specified but --phase6-job-id not provided; skipping Phase 6 trigger")
            else:
                try:
                    import requests
                    
                    # Get Databricks credentials
                    databricks_token = os.environ.get("DATABRICKS_TOKEN", "")
                    workspace_url = os.environ.get("DATABRICKS_HOST", "https://adb-249008710733422.2.azuredatabricks.net")
                    
                    if not workspace_url.startswith("http"):
                        workspace_url = f"https://{workspace_url}"
                    
                    api_url = f"{workspace_url}/api/2.1/jobs/run-now"
                    payload = {
                        "job_id": args.phase6_job_id,
                        "python_params": [
                            "--date", args.date
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
                    print(f"\n✅ Triggered Phase 6 job (ID: {args.phase6_job_id}, Run ID: {run_id})")
                    print(f"   🔗 View run: {workspace_url}/#job/{args.phase6_job_id}/run/{run_id}")
                except Exception as e:
                    print(f"\n⚠️  Failed to trigger Phase 6: {e}")
        
        print("\n" + "=" * 80)
        print("BATCH PREDICTIONS COMPLETE")
        print("=" * 80)
        return 0
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
