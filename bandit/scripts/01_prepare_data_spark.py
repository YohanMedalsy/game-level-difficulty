#!/usr/bin/env python3
"""
Phase 1 (Spark): Feature Selection + Artifacts

Spark-native feature selection using RandomForestClassifier on arm labels to
select top-N features from the combined Delta/Parquet dataset. Writes the
selected_features JSON artifact for downstream phases.

This avoids pandas for large datasets.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import List, Dict

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier, LogisticRegression
from pyspark.ml.regression import RandomForestRegressor, GBTRegressor
from pyspark.ml import Pipeline

# Configure logging
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/dbfs/mnt/bandit/logs/training_pipeline.log'),
        logging.StreamHandler()
    ]
)


def _normalize_dbfs_uri(path: str) -> str:
    """Ensure Spark sees dbfs paths in dbfs:/ form."""
    if path.startswith("/dbfs/"):
        return "dbfs:/" + path[6:]
    return path


def _resolve_local_path(path: str) -> Path:
    """Convert dbfs:/ URIs to /dbfs/... for standard Python file IO."""
    if path.startswith("dbfs:/"):
        return Path("/dbfs" + path[5:])
    return Path(path)


try:
    BANDIT_ROOT = Path(__file__).resolve().parents[1]
except NameError:  # Databricks exec / notebooks don't define __file__
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
ARTIFACTS_DIR = BANDIT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR = BANDIT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Import bandit constants
import sys as _sys
_sys.path.insert(0, str(BANDIT_ROOT / "src"))
from constants import ARM_ORDER


def build_spark() -> SparkSession:
    return (
        SparkSession.builder
        .appName("VW_Bandit_Phase1_Spark")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )


def detect_format(path: str) -> str:
    """
    Detect dataset format (delta vs parquet).

    Local paths: inspect filesystem for _delta_log
    DBFS paths: use naming convention ('.delta' suffix) since Path.exists() cannot see dbfs: URIs.
    """
    if path.startswith("dbfs:/") or path.startswith("/dbfs/"):
        lowered = path.lower()
        if lowered.endswith(".delta") or "_delta_log" in lowered:
            return "delta"
        if lowered.endswith(".parquet"):
            return "parquet"
        # default to delta when suffix ambiguous (most dbfs mounts are delta tables)
        return "delta"

    p = Path(path)
    return "delta" if (p / "_delta_log").exists() else "parquet"


def main():
    parser = argparse.ArgumentParser(description="Spark-native feature selection for VW bandit")
    parser.add_argument('--dataset-path', required=True, help='Path to Delta/Parquet dataset')
    parser.add_argument('--start-date', type=str, default=None, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None, help='End date (YYYY-MM-DD)')
    parser.add_argument('--n-features', type=int, default=50, help='Number of top features to select')
    parser.add_argument('--output-selected-features', type=str, default=str(ARTIFACTS_DIR / 'selected_features_50.json'))
    parser.add_argument('--validate-stability', action='store_true', help='Validate feature stability across seeds')
    parser.add_argument('--stability-seeds', type=str, default='42,43,44,45,46', help='Comma-separated seeds for stability check')
    parser.add_argument('--train-propensity', action='store_true', help='Train Spark ML propensity model and save to models/propensity_spark')
    parser.add_argument('--propensity-model-out', type=str, default=str(MODELS_DIR / 'propensity_spark'), help='Output path for Spark ML propensity model')
    parser.add_argument('--use-train-split-for-propensity', action='store_true', help='Train propensity model on train split only (user-hash 80%)')
    parser.add_argument('--split-seed', type=int, default=42, help='Seed for optional propensity train split (user-hash)')
    # Dual selection controls
    parser.add_argument('--q-model', choices=['rf', 'gbt'], default='rf', help='Model for per-arm Q feature selection')
    parser.add_argument('--q-features-per-arm', type=int, default=20, help='Top-K features to keep per arm for Q selection')
    parser.add_argument('--union-cap', type=int, default=70, help='Cap for weighted union of per-arm features')
    parser.add_argument('--min-rows-per-arm', type=int, default=500, help='Minimum rows required per arm to run Q selection')
    
    # Training pipeline chaining
    parser.add_argument('--trigger-phase2', action='store_true', help='Trigger Phase 2 after Phase 1 completes')
    parser.add_argument('--phase2-job-id', type=int, default=1104819074415450, help='Databricks Job ID for Phase 2')
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

    # Required columns
    required = ['difficulty_arm', 'next_day_reward']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Feature candidates: numeric columns excluding known targets/future
    exclude = {'user_id', 'session_date', 'difficulty_arm', 'next_day_reward',
               'next_effectivelevelmultiplier', 'previous_day_action', 'previous_day_multiplier',
               'action'}
    numeric_types = {'int', 'bigint', 'double', 'float', 'decimal', 'smallint', 'tinyint'}
    dtypes = dict(df.dtypes)
    feature_cols: List[str] = [c for c, t in dtypes.items()
                               if c not in exclude and any(nt in t for nt in numeric_types) and not c.startswith('next_')]

    if not feature_cols:
        raise ValueError("No numeric feature columns found for selection")

    print(f"📊 Candidate numeric features: {len(feature_cols)}")

    # Replace infinities with nulls before fill
    df = df.replace(float("inf"), None)
    df = df.replace(float("-inf"), None)
    # Fill NaNs/nulls in numeric columns to keep Spark ML happy
    if feature_cols:
        df = df.na.fill(0.0, subset=feature_cols)

    # Label encoding for arms
    indexer = StringIndexer(inputCol='difficulty_arm', outputCol='label', handleInvalid='keep')
    df_idx = indexer.fit(df).transform(df)

    assembler = VectorAssembler(inputCols=feature_cols, outputCol='features', handleInvalid='keep')
    df_vec = assembler.transform(df_idx).select('features', 'label')

    # RF classifier to compute importances (baseline selection)
    rf = RandomForestClassifier(labelCol='label', featuresCol='features', numTrees=100, maxDepth=10, seed=42)
    model_rf = rf.fit(df_vec)
    importances_rf = model_rf.featureImportances.toArray().tolist()
    ranked_rf = sorted(zip(feature_cols, importances_rf), key=lambda x: x[1], reverse=True)
    top_n_rf = [name for name, imp in ranked_rf[:args.n_features]]

    # Propensity selection via multinomial LR coefficients (sum |coef| across classes)
    lr_cls = LogisticRegression(featuresCol='features', labelCol='label', family='multinomial', maxIter=50, regParam=1e-4, elasticNetParam=0.0)
    model_lr = lr_cls.fit(df_vec)
    coef_matrix = model_lr.coefficientMatrix.toArray()  # shape: (n_classes, n_features)
    # Aggregate importance per feature
    lr_scores = [float(abs(coef_matrix[:, j]).sum()) for j in range(len(feature_cols))]
    ranked_lr = sorted(zip(feature_cols, lr_scores), key=lambda x: x[1], reverse=True)
    top_n_lr = [name for name, sc in ranked_lr[:args.n_features]]

    print(f"✅ RF-selected top {len(top_n_rf)} features (baseline)")
    for name, imp in ranked_rf[:min(10, len(ranked_rf))]:
        print(f"  • {name:40s} RF {imp:.6f}")
    print(f"✅ LR-selected top {len(top_n_lr)} features (propensity)")
    for name, sc in ranked_lr[:min(10, len(ranked_lr))]:
        print(f"  • {name:40s} LR {sc:.6f}")

    # Save propensity feature selection
    prop_path = ARTIFACTS_DIR / 'selected_features_propensity.json'
    with open(prop_path, 'w') as f:
        json.dump({'selected_features': top_n_lr, 'n_features': len(top_n_lr), 'selection_method': 'spark_lr_coefficients'}, f, indent=2)
    print(f"💾 Saved propensity features: {prop_path}")

    # Per-arm Q selection using RF or GBT regressors
    print("\n🎯 Computing per-arm Q feature selections...")
    per_arm: Dict[str, List[str]] = {}
    counts: Dict[str, int] = {}
    sum_imps: Dict[str, float] = {}

    # Ensure reward column present
    if 'next_day_reward' not in df.columns:
        raise ValueError("Missing 'next_day_reward' for Q selection")

    for arm in ARM_ORDER:
        df_arm = df.filter(F.col('difficulty_arm') == F.lit(arm)).dropna(subset=['next_day_reward'])
        n_rows = df_arm.count()
        if n_rows < args.min_rows_per_arm:
            print(f"⚠️  Arm '{arm}': only {n_rows} rows (<{args.min_rows_per_arm}); skipping Q selection for this arm")
            per_arm[arm] = []
            continue
        # Assemble features
        df_arm_idx = indexer.fit(df_arm).transform(df_arm)
        df_arm_vec = assembler.transform(df_arm_idx).select('features', 'next_day_reward')

        if args.q_model == 'rf':
            q_model = RandomForestRegressor(labelCol='next_day_reward', featuresCol='features', numTrees=150, maxDepth=12, seed=42)
        else:
            q_model = GBTRegressor(labelCol='next_day_reward', featuresCol='features', maxIter=100, maxDepth=6, stepSize=0.1, seed=42)
        try:
            q_fit = q_model.fit(df_arm_vec)
            imps = q_fit.featureImportances.toArray().tolist()
        except Exception as e:
            print(f"⚠️  Arm '{arm}': Q model failed: {e}; skipping")
            per_arm[arm] = []
            continue

        ranked_q = sorted(zip(feature_cols, imps), key=lambda x: x[1], reverse=True)
        top_k = [name for name, imp in ranked_q if imp > 0.0][:args.q_features_per_arm]
        per_arm[arm] = top_k
        # Update weighted union stats
        total_imp = sum(imp for _, imp in ranked_q) or 1.0
        imp_map = {name: imp / total_imp for name, imp in ranked_q if imp > 0.0}
        for name in top_k:
            counts[name] = counts.get(name, 0) + 1
            sum_imps[name] = sum_imps.get(name, 0.0) + imp_map.get(name, 0.0)
        print(f"   Arm '{arm}': selected {len(top_k)} features")

    # Weighted union by (count desc, sum_importance desc), capped
    union_sorted = sorted(counts.keys(), key=lambda n: (counts[n], sum_imps.get(n, 0.0)), reverse=True)
    union_cap = int(args.union_cap)
    top_union = union_sorted[:union_cap]

    # Save artifacts
    q_per_arm_path = ARTIFACTS_DIR / 'selected_features_q_per_arm.json'
    with open(q_per_arm_path, 'w') as f:
        json.dump({'per_arm': per_arm, 'q_model': args.q_model, 'k_per_arm': args.q_features_per_arm}, f, indent=2)
    print(f"💾 Saved per-arm Q features: {q_per_arm_path}")

    union_path = ARTIFACTS_DIR / f'selected_features_union_{union_cap}.json'
    with open(union_path, 'w') as f:
        json.dump({'selected_features': top_union, 'n_features': len(top_union), 'selection_method': 'weighted_union_per_arm', 'counts': counts}, f, indent=2)
    print(f"💾 Saved weighted union features: {union_path}")

    # Back-compat: also save RF baseline file and a 50-cap file for downstream defaults
    out_path = _resolve_local_path(args.output_selected_features)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({'selected_features': top_union[:args.n_features], 'n_features': min(args.n_features, len(top_union)), 'selection_method': 'weighted_union_cap'}, f, indent=2)
    print(f"💾 Saved legacy-cap selection: {args.output_selected_features}")

    print("🎉 Spark feature selection complete")

    # Optional: train Spark ML propensity model on selected features
    if args.train_propensity:
        print("\n🎯 Training Spark ML propensity model (multinomial LogisticRegression)...")
        # Use propensity-selected top-N features
        selected = top_n_lr
        if not selected:
            raise ValueError("No selected features available to train propensity model")

        # Prepare features
        assembler2 = VectorAssembler(inputCols=selected, outputCol='features_prop', handleInvalid='keep')

        # Multinomial logistic regression for multi-class propensities
        lr = LogisticRegression(featuresCol='features_prop', labelCol='label', maxIter=50, regParam=1e-4, elasticNetParam=0.0, family='multinomial')

        stages = [indexer, assembler2, lr]
        pipe = Pipeline(stages=stages)

        df_train_prop = df
        if args.use_train_split_for_propensity:
            print("   Using user-hash 80% train split for propensity training")
            bucket = F.pmod(F.abs(F.hash(F.col('user_id')) + F.lit(args.split_seed)), F.lit(100))
            df_train_prop = df.withColumn('__bucket__', bucket.cast('int'))\
                               .filter(F.col('__bucket__') < F.lit(80))\
                               .drop('__bucket__')

        # Assemble features using the same logic
        df_idx2 = indexer.fit(df_train_prop).transform(df_train_prop)
        df_vec2 = assembler2.transform(df_idx2).select('features_prop', 'label')
        model = lr.fit(df_vec2)

        # Save as PipelineModel-like bundle: we need indexer mapping and assembler config
        # For simplicity, we rebuild a Pipeline and fit on full dataset to persist all params
        pipe_model = Pipeline(stages=[indexer.fit(df), assembler2, model]).fit(df)
        model_out_uri = _normalize_dbfs_uri(args.propensity_model_out)
        pipe_model.write().overwrite().save(model_out_uri)

        local_model_dir = _resolve_local_path(args.propensity_model_out)
        local_model_dir.mkdir(parents=True, exist_ok=True)

        # Save metadata alongside the model directory
        labels = pipe_model.stages[0].labels
        meta = {
            'model_type': 'spark_ml_logistic_regression',
            'labels': labels,
            'n_features': len(selected),
            'features': selected,
            'use_train_split': bool(args.use_train_split_for_propensity),
        }
        meta_path = local_model_dir / 'propensity_spark_meta.json'
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)
        print(f"   ✅ Saved Spark propensity model: {args.propensity_model_out}")
        print(f"   ✅ Metadata: {meta_path}")

    # Optional: stability validation (non-blocking)
    if args.validate_stability:
        print("\n🧪 Validating feature stability across seeds...")
        try:
            seeds = [int(s.strip()) for s in args.stability_seeds.split(',') if s.strip()]
        except Exception:
            seeds = [42, 43, 44, 45, 46]

        ranked_by_seed = []
        for sd in seeds:
            rf_sd = RandomForestClassifier(labelCol='label', featuresCol='features', numTrees=100, maxDepth=10, seed=sd)
            model_sd = rf_sd.fit(df_vec)
            imps_sd = model_sd.featureImportances.toArray().tolist()
            ranked_sd = [name for name, imp in sorted(zip(feature_cols, imps_sd), key=lambda x: x[1], reverse=True)[:args.n_features]]
            ranked_by_seed.append(ranked_sd)

        def jaccard(a, b):
            sa, sb = set(a), set(b)
            inter = len(sa & sb)
            union = len(sa | sb)
            return (inter / union) if union else 1.0

        base = ranked_by_seed[0]
        scores = [jaccard(base, other) for other in ranked_by_seed[1:]]
        avg_jaccard = sum(scores) / len(scores) if scores else 1.0
        print(f"   Stability (avg Jaccard vs seed {seeds[0]}): {avg_jaccard:.3f}")

        stab_path = ARTIFACTS_DIR / 'feature_stability.json'
        with open(stab_path, 'w') as f:
            json.dump({
                'seeds': seeds,
                'n_features': args.n_features,
                'avg_jaccard_vs_first': avg_jaccard,
                'by_seed': [{ 'seed': sd, 'top_features': feats } for sd, feats in zip(seeds, ranked_by_seed)]
            }, f, indent=2)
        print(f"   💾 Saved stability report: {stab_path}")
    
    # Trigger Phase 2 if requested
    if args.trigger_phase2:
        print("\n" + "=" * 80)
        print("TRAINING PIPELINE: TRIGGERING PHASE 2")
        print("=" * 80)
        
        try:
            import requests
            import os
            
            # Get Databricks token
            databricks_token = os.environ.get("DATABRICKS_TOKEN")
            if not databricks_token:
                print("   ⚠️  No DATABRICKS_TOKEN found; skipping Phase 2 trigger")
            else:
                # Get workspace URL
                workspace_url = os.environ.get("DATABRICKS_WORKSPACE_URL", "https://adb-249008710733422.2.azuredatabricks.net")
                if not workspace_url.startswith("http"):
                    workspace_url = f"https://{workspace_url}"
                
                # Trigger Phase 2 job
                api_url = f"{workspace_url}/api/2.1/jobs/run-now"
                
                # Build parameters - only include end-date if specified
                python_params = [
                    "--dataset-path", args.dataset_path,  # Pass through from Phase 0
                    "--output-dir", "dbfs:/mnt/vw/training_vw",
                    "--start-date", args.start_date,  # Pass through
                ]
                
                # Only add end-date if it was specified
                if args.end_date:
                    python_params.extend(["--end-date", args.end_date])
                
                python_params.extend([
                    "--selected-features-json", args.output_selected_features,
                    "--propensity-spark-model", args.propensity_model_out,
                    "--repartition", "256",
                    "--train-ratio", "0.8",
                    "--valid-ratio", "0.1",
                    "--test-ratio", "0.1",
                    "--n-trials", str(args.n_trials)  # Pass through to Phase 3
                ])
                
                payload = {
                    "job_id": args.phase2_job_id,
                    "python_params": python_params
                }
                
                # Retry logic: 1 retry on failure
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
                        
                        print(f"   ✅ Triggered Phase 2 (Job ID: {args.phase2_job_id}, Run ID: {run_id})")
                        logging.info(f"Triggered Phase 2 (Job ID: {args.phase2_job_id}, Run ID: {run_id})")
                        print(f"   🔗 View run: {workspace_url}/#job/{args.phase2_job_id}/run/{run_id}")
                        print(f"   📊 Training Pipeline: Phase 1 → Phase 2 (Convert to VW)")
                        break  # Success, exit retry loop
                        
                    except Exception as retry_error:
                        if attempt < max_retries:
                            print(f"   🔄 Retry {attempt + 1}/{max_retries} after error: {retry_error}")
                            logging.warning(f"Phase 2 trigger attempt {attempt + 1} failed: {retry_error}")
                            print(f"   ⏳ Waiting 5 seconds before retry...")
                            import time
                            time.sleep(5)
                        else:
                            print(f"   ❌ Failed to trigger Phase 2 after {max_retries + 1} attempts: {retry_error}")
                            logging.error(f"Phase 2 trigger failed after {max_retries + 1} attempts: {retry_error}")
                
        except Exception as e:
            print(f"   ⚠️  Failed to trigger Phase 2: {e}")
            import traceback
            traceback.print_exc()


if __name__ == '__main__':
    main()
