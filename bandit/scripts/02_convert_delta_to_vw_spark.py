#!/usr/bin/env python3
"""
Phase 2 (Spark): Delta → VW CB_ADF Streaming Converter

Converts a large Delta/Parquet dataset to Vowpal Wabbit CB-ADF text files
in a distributed, streaming fashion. Avoids materializing the full dataset
on the driver or in pandas.

Defaults:
- Uniform feasible propensities if no propensity model is provided
- Optional sklearn propensity model (Pipeline) via broadcasted bytes

Output: a directory of sharded CB-ADF text files suitable for VW training.

Example:
  python bandit/scripts/02_convert_delta_to_vw_spark.py \
    --dataset-path ../test_folder/daily_features_spark.delta \
    --start-date 2025-10-01 \
    --end-date 2025-10-28 \
    --output-dir bandit/data/processed/vw_streaming \
    --selected-features-json bandit/artifacts/selected_features_50.json \
    --repartition 512

With propensity model (optional):
  python bandit/scripts/02_convert_delta_to_vw_spark.py \
    --dataset-path ../test_folder/daily_features_spark.delta \
    --output-dir bandit/data/processed/vw_streaming \
    --selected-features-json bandit/artifacts/selected_features_50.json \
    --propensity-model bandit/models/propensity_model.pkl
"""

import sys
import argparse
import json
import pickle
import shutil
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml import Pipeline, PipelineModel

# Ensure log directory exists
log_path = '/dbfs/mnt/bandit/logs/training_pipeline.log'
try:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
except Exception:
    pass

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)
# Utility helpers for DBFS/local interoperability
def _normalize_dbfs_uri(path: str) -> str:
    """Ensure dbfs paths use the dbfs:/ scheme."""
    if path.startswith("/dbfs/"):
        return "dbfs:/" + path[6:]
    return path


def _to_local_path(path: str) -> Path:
    """Convert dbfs:/ URIs to /dbfs/... local paths for Python IO."""
    if path.startswith("dbfs:/"):
        return Path("/dbfs" + path[5:])
    if path.startswith("/dbfs/"):
        return Path(path)
    return Path(path)


def _remove_path_if_exists(path: str) -> None:
    """Best-effort removal of existing output directories on DBFS or local FS."""
    normalized = _normalize_dbfs_uri(path)
    # Try DBFS removal first
    try:
        dbutils_instance = globals().get("dbutils")
        if dbutils_instance is not None:
            dbutils_instance.fs.rm(normalized, True)  # type: ignore[attr-defined]
            return
    except Exception:
        pass

    # Fallback to local filesystem
    local_path = _to_local_path(path)
    try:
        if local_path.exists():
            shutil.rmtree(local_path)
    except Exception:
        pass


# Ensure we can import project constants even when __file__ is undefined (Databricks exec)
try:
    BANDIT_ROOT = Path(__file__).resolve().parents[1]
except NameError:  # pragma: no cover
    import inspect

    file_path = None
    frame = inspect.currentframe()
    while frame and not file_path:
        candidate = frame.f_code.co_filename
        if candidate and candidate not in {"<stdin>", "<string>"}:
            file_path = Path(candidate).resolve()
        frame = frame.f_back

    if file_path and len(file_path.parents) >= 2:
        BANDIT_ROOT = file_path.parents[1]
    else:
        cwd = Path.cwd().resolve()
        if cwd.name == "scripts" and (cwd.parent / "bandit").exists():
            BANDIT_ROOT = cwd.parent
        elif (cwd / "bandit").exists():
            BANDIT_ROOT = cwd / "bandit"
        else:
            BANDIT_ROOT = cwd

sys.path.insert(0, str(BANDIT_ROOT / "src"))
from constants import (
    ACTION_TO_DELTA,
    ARM_ORDER,
    MIN_MULTIPLIER,
    MAX_MULTIPLIER,
    assign_feature_to_namespace,
)


def build_spark(app_name: str = "VW_Delta_to_ADF") -> SparkSession:
    return (
        SparkSession.builder
        .appName(app_name)
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def detect_format(path: str) -> str:
    """Detect whether the dataset directory is Delta or Parquet."""
    if path.startswith("dbfs:/") or path.startswith("/dbfs/"):
        lowered = path.lower()
        if lowered.endswith(".delta") or "_delta_log" in lowered:
            return "delta"
        if lowered.endswith(".parquet"):
            return "parquet"
        # default to delta when suffix ambiguous (most dbfs mounts are delta tables)
        return "delta"

    p = Path(path)
    if (p / "_delta_log").exists():
        return "delta"
    return "parquet"


def load_selected_features(json_path: str) -> Optional[List[str]]:
    if not json_path:
        return None
    p = _to_local_path(json_path)
    if not p.exists():
        print(f"⚠️  Selected features JSON not found at: {json_path}")
        return None
    try:
        with open(p, 'r') as f:
            data = json.load(f)
        feats = data.get('selected_features')
        if isinstance(feats, list) and feats:
            cleaned = [f for f in feats if f != "action"]
            if len(cleaned) != len(feats):
                print("ℹ️  Dropped leakage-prone feature 'action' from selected features list")
            return cleaned
    except Exception as exc:
        print(f"⚠️  Failed to load selected features from {json_path}: {exc}")
    return None


def load_propensity_model_bytes(path: Optional[str]) -> Optional[bytes]:
    if not path:
        return None
    p = _to_local_path(path)
    if not p.exists():
        print(f"ℹ️  sklearn propensity pickle not found at: {path}")
        return None
    if p.is_dir():
        print(f"ℹ️  Skipping sklearn propensity load because path is a directory: {path}")
        return None
    with open(p, 'rb') as f:
        return f.read()


def load_propensity_spark_meta(path: str) -> Optional[Dict[str, any]]:
    meta_path = _to_local_path(path) / 'propensity_spark_meta.json'
    if not meta_path.exists():
        print(f"ℹ️  Propensity Spark metadata not found at: {meta_path}")
        return None
    try:
        with open(meta_path, 'r') as f:
            return json.load(f)
    except Exception as exc:
        print(f"⚠️  Failed to load propensity Spark metadata: {exc}")
        return None


def to_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def make_feasible_arms(current_mult: float) -> List[str]:
    feas = []
    for arm, delta in ACTION_TO_DELTA.items():
        nm = current_mult + delta
        if MIN_MULTIPLIER <= nm <= MAX_MULTIPLIER:
            feas.append(arm)
    return feas


def build_namespace_plan(features: List[str]) -> Dict[str, List[str]]:
    ns: Dict[str, List[str]] = {}
    for f in features:
        ns_id = assign_feature_to_namespace(f)
        ns.setdefault(ns_id, []).append(f)
    return ns


def format_shared_line(
    row_dict: Dict[str, any],
    features: List[str],
    ns_plan: Dict[str, List[str]],
    normalization_stats: Optional[Dict[str, Tuple[float, float]]] = None
) -> str:
    parts = []
    for ns, feats in sorted(ns_plan.items()):
        ns_feats = []
        for f in feats:
            raw_val = to_float(row_dict.get(f, 0.0))
            
            # Apply normalization if stats exist
            val = raw_val
            if normalization_stats and f in normalization_stats:
                min_v, max_v = normalization_stats[f]
                if max_v > min_v:
                    val = (raw_val - min_v) / (max_v - min_v)
                    # Clip to [0, 1] to handle outliers in Test set that exceed Train bounds
                    # This ensures strict magnitude control for VW interactions
                    val = max(0.0, min(1.0, val))
                else:
                    val = 0.0  # Constant feature -> 0.0
            
            if val != 0.0:
                clean = f.replace(' ', '_').replace(':', '_').replace('|', '_')
                ns_feats.append(f"{clean}:{val:.6f}")
        if ns_feats:
            parts.append(f"|{ns} " + " ".join(ns_feats))
    shared = "shared " + " ".join(parts)
    # Append user_id as a comment for alignment (ignored by VW training)
    uid = row_dict.get('user_id')
    if uid is not None:
        try:
            uid_val = int(uid)
            shared = shared + f"  # uid:{uid_val}"
        except Exception:
            pass
    return shared


def build_converter(
    selected_features: Optional[List[str]],
    propensity_model_bytes: Optional[bytes],
    normalization_stats: Optional[Dict[str, Tuple[float, float]]] = None,
    propensity_col: Optional[str] = None
) -> Tuple:
    """Create a partition mapper closure to format CB-ADF lines."""

    # Prepare namespace plan on driver
    if selected_features:
        feat_list = selected_features
    else:
        feat_list = []  # Will produce minimal shared line

    ns_plan = build_namespace_plan(feat_list) if feat_list else {}

    def map_partition(iter_rows):
        # Lazy import to executors
        import numpy as np
        import pickle

        model = None
        classes = None
        if propensity_model_bytes is not None:
            try:
                model = pickle.loads(propensity_model_bytes)
                if hasattr(model, 'classes_'):
                    classes = list(model.classes_)
            except Exception:
                model = None
                classes = None

        def predict_proba(xvec: np.ndarray) -> List[float]:
            if model is None:
                return []
            try:
                probs = model.predict_proba(xvec.reshape(1, -1))[0]
                if classes is not None and set(classes) == set(ARM_ORDER):
                    arm_to_prob = {arm: 0.0 for arm in ARM_ORDER}
                    for i, arm in enumerate(classes):
                        arm_to_prob[arm] = float(probs[i])
                    return [arm_to_prob[a] for a in ARM_ORDER]
                else:
                    return [float(p) for p in probs[:len(ARM_ORDER)]]
            except Exception:
                return []

        for r in iter_rows:
            rd = r.asDict(recursive=True)
            current_mult = to_float(rd.get('current_effectivelevelmultiplier'))
            reward = to_float(rd.get('next_day_reward'))
            chosen_arm = rd.get('difficulty_arm')
            if chosen_arm not in ACTION_TO_DELTA:
                continue

            feas = make_feasible_arms(current_mult)
            shared = format_shared_line(rd, feat_list, ns_plan, normalization_stats)

            # Determine propensities
            probs = None
            
            # 1. Use logged propensity if column is provided and valid
            if propensity_col:
                logged_p = rd.get(propensity_col)
                chosen_idx = ARM_ORDER.index(chosen_arm) if chosen_arm in ARM_ORDER else -1
                
                if chosen_idx >= 0 and isinstance(logged_p, (int, float)):
                    # Initialize with 0.0 or uniform for non-chosen? 
                    # VW expects P(chosen_action) to be correct. For OPE/training, only the chosen action's prob matters strictly,
                    # but having a full distribution is better.
                    # Strategy: Set chosen prob, then distribute remainder uniformly among other FEASIBLE arms?
                    # Simpler strategy often used: Only the chosen probability is strictly required in the label.
                    # The other probabilities are informative.
                    # Let's set the chosen probability exactly.
                    # For the others, we can leave them 0 or uniform remainder. 
                    # Existing logic was using model or uniform. 
                    
                    # We will construct a probability vector where chosen gets logged_p
                    # If logged_p < 1.0, we can distribute 1-logged_p among other FEASIBLE arms.
                    
                    probs = [0.0] * len(ARM_ORDER)
                    probs[chosen_idx] = float(logged_p)
                    
                    # Distribute remainder
                    remainder = 1.0 - probs[chosen_idx]
                    if remainder > 0:
                        other_feas = [a for a in feas if a != chosen_arm]
                        if other_feas:
                            val = remainder / len(other_feas)
                            for a in other_feas:
                                idx = ARM_ORDER.index(a)
                                probs[idx] = val
                
            # 2. If no valid logged propensity, try model
            if probs is None and model is not None and feat_list:
                x = np.array([to_float(rd.get(f, 0.0)) for f in feat_list], dtype=float)
                probs = predict_proba(x)

            # 3. Fallback: Uniform over feasible
            if not probs or len(probs) != len(ARM_ORDER):
                probs = [0.0] * len(ARM_ORDER)
                if len(feas) > 0:
                    u = 1.0 / len(feas)
                    for i, arm in enumerate(ARM_ORDER):
                        probs[i] = u if arm in feas else 0.0
                else:
                    probs = [1.0 / len(ARM_ORDER)] * len(ARM_ORDER)
            else:
                # Mask infeasible and normalize
                for i, arm in enumerate(ARM_ORDER):
                    if arm not in feas:
                        probs[i] = 0.0
                total = sum(probs)
                if total > 0:
                    probs = [p / total for p in probs]
                else:
                    # Fallback if masking killed all probability mass
                    probs = [0.0] * len(ARM_ORDER)
                    if len(feas) > 0:
                        u = 1.0 / len(feas)
                        for i, arm in enumerate(ARM_ORDER):
                            probs[i] = u if arm in feas else 0.0
                    else:
                        probs = [1.0 / len(ARM_ORDER)] * len(ARM_ORDER)

            lines = [shared]
            for idx, arm in enumerate(ARM_ORDER):
                delta = ACTION_TO_DELTA[arm]
                feasible = 1 if arm in feas else 0
                action_features = f"|a arm:{idx} delta:{delta:.2f} |c mult:{current_mult:.2f} feasible:{feasible}"

                if arm == chosen_arm:
                    cost = -reward
                    prob = probs[idx]
                    label_prefix = f"{idx}:{cost:.4f}:{prob:.6f} "
                else:
                    label_prefix = ""

                action_line = f"{label_prefix}{action_features}"
                lines.append(action_line)
            yield "\n".join(lines) + "\n"

    return map_partition


def main():
    parser = argparse.ArgumentParser(description="Delta → VW CB-ADF streaming converter (Spark)")
    parser.add_argument('--dataset-path', required=True, help='Path to Delta/Parquet dataset directory')
    parser.add_argument('--output-dir', required=True, help='Output directory root for CB-ADF shards (splits written under train/, valid/, test/)')
    parser.add_argument('--start-date', type=str, default=None, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None, help='End date (YYYY-MM-DD)')
    parser.add_argument('--selected-features-json', type=str, default='', help='Path to JSON with selected_features list')
    parser.add_argument('--propensity-model', type=str, default='', help='Optional sklearn Pipeline pickle for propensities (legacy)')
    parser.add_argument('--propensity-spark-model', type=str, default=str(BANDIT_ROOT / 'models' / 'propensity_spark'), help='Path to Spark ML propensity PipelineModel')
    parser.add_argument('--propensity-col', type=str, default='', help='Optional column name containing logged propensity for chosen action')
    parser.add_argument('--repartition', type=int, default=256, help='Number of output shards per split (repartition)')
    parser.add_argument('--shard-suffix', type=str, default='', help='Optional suffix appended to output directory name for this run')
    parser.add_argument('--train-ratio', type=float, default=0.8, help='Train split ratio (0-1)')
    parser.add_argument('--valid-ratio', type=float, default=0.1, help='Validation split ratio (0-1)')
    parser.add_argument('--test-ratio', type=float, default=0.1, help='Test split ratio (0-1)')
    parser.add_argument('--train-split', type=float, default=None, help='Alias for --train-ratio')
    parser.add_argument('--valid-split', type=float, default=None, help='Alias for --valid-ratio')
    parser.add_argument('--test-split', type=float, default=None, help='Alias for --test-ratio (defaults to 1 - train - valid)')
    parser.add_argument('--split-seed', type=int, default=42, help='Deterministic split seed (hash salt)')
    
    # Training pipeline chaining
    parser.add_argument('--trigger-phase3', action='store_true', help='Trigger Phase 3 after Phase 2 completes')
    
    # Defaults from env vars or fallback to hardcoded (for now, until config system is fully adopted)
    default_job_id = int(os.environ.get("PHASE3_JOB_ID", "353591756279168"))
    parser.add_argument('--phase3-job-id', type=int, default=default_job_id, help='Databricks Job ID for Phase 3')
    parser.add_argument('--n-trials', type=int, default=100, help='Number of Optuna trials for Phase 3 (pass-through parameter)')
    
    args = parser.parse_args()
    
    # Validate n-trials
    if not 2 <= args.n_trials <= 300:
        raise ValueError(f"--n-trials must be between 2 and 300, got {args.n_trials}")

    spark = build_spark()
    spark.sparkContext.setLogLevel("WARN")

    fmt = detect_format(args.dataset_path)
    print(f"📂 Reading dataset: {args.dataset_path} ({fmt})")
    df = spark.read.format(fmt).load(args.dataset_path)

    if args.start_date:
        df = df.filter(F.col('session_date') >= F.lit(args.start_date))
    if args.end_date:
        df = df.filter(F.col('session_date') <= F.lit(args.end_date))

    # Filter out negative rewards (outliers)
    # This is critical as VW minimizes cost = -reward. A large negative reward becomes a large positive cost,
    # skewing the loss metric.
    df = df.filter(F.col('next_day_reward') >= 0)
    print("🧹 Filtered out negative rewards (outliers)")

    # ---------------------------------------------------------
    # Robust Outlier Capping (Winsorization) at P99.9
    # ---------------------------------------------------------
    # We calculate the P99.9 threshold on the *entire batch* (pre-split).
    # This effectively treats extreme outliers as data artifacts to be cleaned
    # based on the distribution of the current data batch.
    try:
        # approxQuantile(col, probabilities, relativeError)
        # 0.001 relative error is sufficient for this purpose
        thresholds = df.approxQuantile('next_day_reward', [0.999], 0.001)
        if thresholds:
            p999_cap = float(thresholds[0])
            print(f"📊 Winsorization: Capping 'next_day_reward' at P99.9 ({p999_cap:.4f})")
            
            df = df.withColumn(
                'next_day_reward',
                F.when(F.col('next_day_reward') > p999_cap, p999_cap)
                 .otherwise(F.col('next_day_reward'))
            )
        else:
            print("⚠️  Winsorization skipped: approxQuantile returned empty result.")
    except Exception as e:
        print(f"⚠️  Winsorization failed: {e}. Proceeding with raw rewards.")

    required = ['user_id', 'session_date', 'difficulty_arm', 'next_day_reward', 'current_effectivelevelmultiplier']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    selected_features = load_selected_features(args.selected_features_json)
    if selected_features:
        print(f"✅ Using selected features: {len(selected_features)}")
        feature_cols = selected_features
    else:
        print("⚠️  No selected features provided. Proceeding with minimal shared line (no features).")
        feature_cols = []

    # Include propensity column if provided
    prop_col = args.propensity_col if args.propensity_col else None
    if prop_col and prop_col not in df.columns:
        print(f"⚠️  Propensity column '{prop_col}' not found; ignoring.")
        prop_col = None

    # Alias handling for split ratios
    train_ratio = args.train_split if args.train_split is not None else args.train_ratio
    valid_ratio = args.valid_split if args.valid_split is not None else args.valid_ratio
    if args.test_split is not None:
        test_ratio = args.test_split
    else:
        # Recompute test ratio if aliases supplied and user didn't explicitly set test
        if (args.train_split is not None) or (args.valid_split is not None):
            test_ratio = 1.0 - train_ratio - valid_ratio
        else:
            test_ratio = args.test_ratio

    if test_ratio < 0:
        if abs(test_ratio) < 1e-8:
            test_ratio = 0.0
        else:
            raise ValueError(f"Computed test split ratio is negative ({test_ratio}); adjust train/valid splits.")

    # Validate split ratios
    r_sum = train_ratio + valid_ratio + test_ratio
    if abs(r_sum - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0 (got {r_sum})")

    propensity_meta = None
    propensity_feature_cols: List[str] = []
    if args.propensity_spark_model:
        propensity_meta = load_propensity_spark_meta(args.propensity_spark_model)
        if propensity_meta:
            propensity_feature_cols = propensity_meta.get('features', [])
            missing_prop_cols = [c for c in propensity_feature_cols if c not in df.columns]
            if missing_prop_cols:
                print(f"⚠️  Propensity features missing from dataset and will be ignored: {missing_prop_cols[:10]}")
            propensity_feature_cols = [c for c in propensity_feature_cols if c in df.columns]
            if propensity_feature_cols:
                print(f"✅ Including {len(propensity_feature_cols)} propensity feature columns required by Spark model")

    # Select needed columns only
    keep_cols = list(set(required + feature_cols + propensity_feature_cols + ([prop_col] if prop_col else [])))
    df_keep = df.select(*[c for c in keep_cols if c in df.columns])

    # If no explicit propensity column and a Spark ML model path exists, compute logged_propensity via Spark ML
    if (prop_col is None) and args.propensity_spark_model and _to_local_path(args.propensity_spark_model).exists():
        try:
            from pyspark.ml.pipeline import PipelineModel
            from pyspark.sql.types import DoubleType
            from pyspark.sql import functions as SF
            print(f"⚙️  Loading Spark ML propensity model: {args.propensity_spark_model}")
            pm = PipelineModel.load(_normalize_dbfs_uri(args.propensity_spark_model))

            # Defensive: verify label mapping vs expected arms
            labels_from_model = None
            try:
                for st in pm.stages:
                    if hasattr(st, 'labels') and getattr(st, 'labels') is not None:
                        labels_from_model = list(getattr(st, 'labels'))
                        break
            except Exception:
                labels_from_model = None
            if labels_from_model is not None:
                if set(labels_from_model) != set(ARM_ORDER):
                    print(f"⚠️  Spark model label set {labels_from_model} differs from ARM_ORDER {ARM_ORDER}. Proceeding; chosen-arm index remains correct via StringIndexer.")
                elif labels_from_model != ARM_ORDER:
                    print(f"ℹ️  Spark model label order {labels_from_model} != ARM_ORDER {ARM_ORDER}. This is OK; we extract the chosen-arm probability by its indexed label.")
            scored = pm.transform(df_keep)
            # 'label' corresponds to difficulty_arm index, 'probability' is a vector
            vector_get = SF.udf(lambda v, i: float(v[i]) if v is not None and i is not None and int(i) < len(v) and int(i) >= 0 else None, DoubleType())
            scored = scored.withColumn('label_int', SF.col('label').cast('int'))
            scored = scored.withColumn('logged_propensity', vector_get(SF.col('probability'), SF.col('label_int')))
            df_keep = scored.select(*[c for c in df_keep.columns], 'logged_propensity')
            prop_col = 'logged_propensity'
            print("   ✅ Computed logged_propensity via Spark ML model")
        except Exception as e:
            print(f"⚠️  Failed to compute Spark ML propensities: {e}. Falling back to uniform feasible.")

    # Fill NaNs/nulls in feature columns with 0.0 BEFORE splitting
    # This ensures consistency and prevents nulls from affecting stats computation or conversion
    if feature_cols:
        df_keep = df_keep.na.fill(0.0, subset=feature_cols)

    # Deterministic user-based split using hashed user_id bucket (0-99)
    # Ensures all rows for a user land in the same split and is reproducible
    bucket = F.pmod(F.abs(F.hash(F.col('user_id')) + F.lit(args.split_seed)), F.lit(100))
    df_split = df_keep.withColumn('__bucket__', bucket.cast('int'))

    train_upper = int(round(train_ratio * 100))
    valid_upper = train_upper + int(round(valid_ratio * 100))

    df_train = df_split.filter(F.col('__bucket__') < F.lit(train_upper)).drop('__bucket__')
    df_valid = df_split.filter((F.col('__bucket__') >= F.lit(train_upper)) & (F.col('__bucket__') < F.lit(valid_upper))).drop('__bucket__')
    df_test  = df_split.filter(F.col('__bucket__') >= F.lit(valid_upper)).drop('__bucket__')

    # Coverage summary (early warning)
    def _print_counts(label: str, sdf):
        try:
            counts = sdf.groupBy('difficulty_arm').count().toPandas().sort_values('difficulty_arm')
            print(f"\n📊 {label} split coverage:")
            for _, row in counts.iterrows():
                print(f"   {row['difficulty_arm']}: {int(row['count'])}")
        except Exception as e:
            print(f"⚠️  Failed to print {label} counts: {e}")
    _print_counts('Train', df_train)
    _print_counts('Valid', df_valid)
    _print_counts('Test', df_test)

    model_bytes = load_propensity_model_bytes(args.propensity_model)
    if model_bytes:
        print("✅ sklearn propensity model provided; probabilities will be model-based (masked by feasibility).")
    elif prop_col:
        print("✅ Using Spark ML logged_propensity for chosen action; non-chosen lines use uniform over feasible.")
    else:
        print("ℹ️  No propensity model; using uniform probabilities over feasible arms.")

    # Compute normalization stats (MinMax) for selected features
    normalization_stats = {}
    if feature_cols:
        print(f"📊 Computing normalization stats for {len(feature_cols)} features (using TRAIN split only)...")
        
        # Compute min/max in one pass on df_train to avoid leakage
        aggs = []
        for c in feature_cols:
            aggs.append(F.min(c).alias(f"min_{c}"))
            aggs.append(F.max(c).alias(f"max_{c}"))
        
        try:
            stats_row = df_train.select(aggs).first()
            if stats_row:
                stats_dict = stats_row.asDict()
                for c in feature_cols:
                    min_v = float(stats_dict.get(f"min_{c}") or 0.0)
                    max_v = float(stats_dict.get(f"max_{c}") or 0.0)
                    normalization_stats[c] = (min_v, max_v)
                print("   ✅ Stats computed")
        except Exception as e:
            print(f"⚠️  Failed to compute normalization stats: {e}. Proceeding without normalization.")

    # Wrap map function to inject chosen-propensity column if present
    base_map_fn = build_converter(selected_features, model_bytes, normalization_stats, prop_col)

    def map_with_propensity(iter_rows):
        for line in base_map_fn(iter_rows):
            yield line

    map_fn = map_with_propensity

    out_dir = args.output_dir.rstrip('/').rstrip('\\')
    if args.shard_suffix:
        out_dir = f"{out_dir}_{args.shard_suffix}"
    
    # Define paths for VW text output
    vw_train_path = str(Path(out_dir) / 'train')
    vw_valid_path = str(Path(out_dir) / 'valid')
    vw_test_path = str(Path(out_dir) / 'test')

    # Define paths for Delta table output (for Phase 4 validation)
    delta_train_path = str(Path(out_dir) / 'train.delta')
    delta_valid_path = str(Path(out_dir) / 'valid.delta')
    delta_test_path = str(Path(out_dir) / 'test.delta')

    # Cleanup existing
    for p in (vw_train_path, vw_valid_path, vw_test_path, delta_train_path, delta_valid_path, delta_test_path):
        _remove_path_if_exists(p)

    print(f"💾 Writing CB-ADF shards to: {out_dir}/{{train,valid,test}}")
    print(f"💾 Writing Delta splits to: {out_dir}/{{train,valid,test}}.delta")

    # Write Train
    df_train.repartition(args.repartition).rdd.mapPartitions(map_fn).saveAsTextFile(vw_train_path)
    df_train.write.format("delta").mode("overwrite").save(delta_train_path)

    # Write Valid
    df_valid.repartition(max(1, args.repartition // 4)).rdd.mapPartitions(map_fn).saveAsTextFile(vw_valid_path)
    df_valid.write.format("delta").mode("overwrite").save(delta_valid_path)

    # Write Test
    df_test.repartition(max(1, args.repartition // 4)).rdd.mapPartitions(map_fn).saveAsTextFile(vw_test_path)
    df_test.write.format("delta").mode("overwrite").save(delta_test_path)

    # Save Metadata
    metadata = {
        "dataset_path": args.dataset_path,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "split_seed": args.split_seed,
        "ratios": {"train": train_ratio, "valid": valid_ratio, "test": test_ratio},
        "paths": {
            "vw_train": _normalize_dbfs_uri(vw_train_path),
            "vw_valid": _normalize_dbfs_uri(vw_valid_path),
            "vw_test": _normalize_dbfs_uri(vw_test_path),
            "delta_train": _normalize_dbfs_uri(delta_train_path),
            "delta_valid": _normalize_dbfs_uri(delta_valid_path),
            "delta_test": _normalize_dbfs_uri(delta_test_path),
        }
    }
    
    # Ensure local path for metadata exists
    # If out_dir is a dbfs:/ URI, we must convert to /dbfs/... for local IO
    local_out_dir = _to_local_path(out_dir)
    os.makedirs(local_out_dir, exist_ok=True)
    
    meta_path = local_out_dir / "split_metadata.json"
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Saved split metadata to: {meta_path}")

    print("✅ Delta → CB-ADF conversion complete (train/valid/test)")
    
    # Trigger Phase 3 if requested
    if args.trigger_phase3:
        print("\n" + "=" * 80)
        print("TRAINING PIPELINE: TRIGGERING PHASE 3")
        print("=" * 80)
        
        try:
            import requests
            
            # Get Databricks token
            databricks_token = os.environ.get("DATABRICKS_TOKEN")
            if not databricks_token:
                print("   ⚠️  No DATABRICKS_TOKEN found; skipping Phase 3 trigger")
            else:
                # Get workspace URL
                workspace_url = os.environ.get("DATABRICKS_WORKSPACE_URL", "https://adb-249008710733422.2.azuredatabricks.net")
                if not workspace_url.startswith("http"):
                    workspace_url = f"https://{workspace_url}"
                
                # Trigger Phase 3 job
                api_url = f"{workspace_url}/api/2.1/jobs/run-now"
                payload = {
                    "job_id": args.phase3_job_id,
                    "python_params": [
                        "--streaming-dir", args.output_dir,
                        "--n-trials", str(args.n_trials),
                        "--n-jobs", "1"
                    ]
                }
                
                # Retry logic: 1 retry on failure
                import logging
                max_retries = 1
                for attempt in range(max_retries + 1):
                    try:
                        response = requests.post(
                            api_url,
                            headers={"Authorization": f"Bearer {databricks_token}"},
                            json=payload,
                            timeout=10
                        )
                        response.raise_for_status()
                        run_id = response.json().get("run_id")
                        
                        print(f"   ✅ Triggered Phase 3 (Job ID: {args.phase3_job_id}, Run ID: {run_id})")
                        logging.info(f"Triggered Phase 3 (Job ID: {args.phase3_job_id}, Run ID: {run_id})")
                        print(f"   🔗 View run: {workspace_url}/#job/{args.phase3_job_id}/run/{run_id}")
                        print(f"   📊 Training Pipeline: Phase 2 → Phase 3 (Train Model)")
                        break  # Success, exit retry loop
                        
                    except Exception as retry_error:
                        if attempt < max_retries:
                            print(f"   🔄 Retry {attempt + 1}/{max_retries} after error: {retry_error}")
                            logging.warning(f"Phase 3 trigger attempt {attempt + 1} failed: {retry_error}")
                            print(f"   ⏳ Waiting 5 seconds before retry...")
                            import time
                            time.sleep(5)
                        else:
                            print(f"   ❌ Failed to trigger Phase 3 after {max_retries + 1} attempts: {retry_error}")
                            logging.error(f"Phase 3 trigger failed after {max_retries + 1} attempts: {retry_error}")
                
        except Exception as e:
            print(f"   ⚠️  Failed to trigger Phase 3: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
