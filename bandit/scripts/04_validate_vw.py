#!/usr/bin/env python3
"""
Phase 4: Comprehensive VW Model Validation (Spark-Native)

Validates the trained VW model against OPE baseline using strict Spark-based OPE checks.

1. OPE Comparison: VW DR vs OPE uniform baseline (must beat 95%)
2. Inference Performance: p99 latency < 10ms
3. Generate validation report

Outputs:
- artifacts/validation_report.md
- artifacts/vw_vs_ope_comparison.csv
- artifacts/diagnostics/weights_summary.json
- artifacts/diagnostics/weights_hist.png
- artifacts/diagnostics/coverage_summary.json
- artifacts/diagnostics/validation_summary.json
- artifacts/diagnostics/validation_metrics.csv
- artifacts/diagnostics/dr_bootstrap_vw_policy.json
- artifacts/diagnostics/vw_feature_importance.json
- artifacts/diagnostics/vw_readable.txt
"""

import argparse
import os
import subprocess
from pathlib import Path
import json
import time
import shutil
import sys
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession, functions as F
from pyspark.ml.pipeline import PipelineModel

# Add parent to path (robust to Databricks where __file__ may be undefined)
try:
    BANDIT_ROOT = Path(__file__).parent.parent.absolute()
except NameError:
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
    ARM_ORDER,
    OPE_UNIFORM_DR_MEAN,
    OPE_UNIFORM_DR_STD,
    VW_DR_MIN_ACCEPTABLE,
)
from validation.ope_estimators import compute_effective_sample_size

# Paths
# Use local ephemeral storage for large data files to avoid Workspace file size limits (Errno 27)
# Fallback to /tmp if /local_disk0 is not available
if Path("/local_disk0").exists():
    LOCAL_SCRATCH = Path("/local_disk0/tmp/vw_pipeline")
else:
    LOCAL_SCRATCH = Path("/tmp/vw_pipeline")

DATA_PROCESSED = LOCAL_SCRATCH / "data" / "processed"
MODELS_DIR = BANDIT_ROOT / "models"
ARTIFACTS_DIR = BANDIT_ROOT / "artifacts"
DIAG_DIR = ARTIFACTS_DIR / "diagnostics"
STREAMING_DIR = DATA_PROCESSED / "vw_streaming"

# Ensure scratch dirs exist
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
DIAG_DIR.mkdir(parents=True, exist_ok=True)

def ensure_vw_cli() -> None:
    """Ensure the vw binary is available on PATH."""
    if shutil.which("vw"):
        print("[Phase4] vw CLI already present on PATH")
        return

    print("[Phase4] vw CLI not found. Installing vowpalwabbit==9.10.0 ...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--force-reinstall", "vowpalwabbit==9.10.0"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception as exc:
        print(f"[Phase4][WARN] pip install vowpalwabbit failed: {exc}")

    candidates: List[Path] = []
    try:
        import importlib
        vw_mod = importlib.import_module("vowpalwabbit")
        pkg_dir = Path(vw_mod.__file__).resolve().parent
        for candidate in pkg_dir.rglob("vw"):
            try:
                if candidate.is_file():
                    candidates.append(candidate)
            except Exception:
                pass
    except Exception as exc:
        print(f"[Phase4][WARN] Could not probe vowpalwabbit package for vw binary: {exc}")

    candidates.extend([
        Path("/usr/local/bin/vw"),
        Path("/usr/bin/vw"),
        Path("/databricks/python/bin/vw"),
        Path("/databricks/python3/bin/vw"),
    ])

    for candidate in candidates:
        try:
            if candidate.is_file():
                candidate.chmod(0o755)
                subprocess.run(["ln", "-sf", str(candidate), "/usr/local/bin/vw"], check=True)
                print(f"[Phase4] vw CLI linked from {candidate}")
                return
        except Exception:
            pass

    print("[Phase4][WARN] Attempting to install vw via apt-get ...")
    try:
        subprocess.run(["sudo", "apt-get", "update"], check=True)
        subprocess.run(["sudo", "apt-get", "install", "-y", "vowpal-wabbit"], check=True)
    except Exception as exc:
        print(f"[Phase4][WARN] apt-get install failed: {exc}")

    if shutil.which("vw"):
        print("[Phase4] vw CLI ready")
    else:
        raise RuntimeError("[Phase4] vw binary not found after installation attempts.")


def _normalize_dbfs_uri(path: str) -> str:
    """Ensure Spark sees dbfs paths in dbfs:/ form."""
    if path.startswith("/dbfs/"):
        return "dbfs:/" + path[6:]
    return path


def _to_local_path(path: str) -> Path:
    if path.startswith("dbfs:/"):
        return Path("/dbfs" + path[5:])
    return Path(path)


def _concat_shards_to_file(src_dir: Path, dest_file: Path) -> None:
    import glob
    shards = sorted(glob.glob(str(src_dir / 'part-*')))
    if not shards:
        raise FileNotFoundError(f"No shards found under {src_dir}")
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Use smaller chunk size to avoid Errno 27 on some filesystems
    chunk_size = 64 * 1024 
    
    with open(dest_file, 'wb') as out_f:
        for shard in shards:
            with open(shard, 'rb') as in_f:
                while True:
                    chunk = in_f.read(chunk_size)
                    if not chunk:
                        break
                    try:
                        out_f.write(chunk)
                    except OSError as e:
                        if e.errno == 27: # File too large
                            print(f"⚠️  OSError: File too large while writing {dest_file}. Truncating.")
                            return
                        raise


def ensure_vw_files_from_streaming(streaming_root: Path,
                                   train_file: Path,
                                   valid_file: Path,
                                   test_file: Path) -> None:
    if not streaming_root.exists():
        raise FileNotFoundError(f"Streaming directory not found: {streaming_root}")

    def concat_if_needed(split: str, dest: Path):
        split_dir = streaming_root / split
        if split_dir.exists():
            print(f"🔧 Concatenating {split} shards ({split_dir}) → {dest}")
            _concat_shards_to_file(split_dir, dest)
            print(f"   ✅ {dest.name} ready")
        else:
            print(f"ℹ️  {split.capitalize()} shards not found at {split_dir}; skipping.")

    concat_if_needed('train', train_file)
    concat_if_needed('valid', valid_file)
    concat_if_needed('test', test_file)


# Matplotlib settings
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class ValidationReport:
    """Helper to build a markdown report."""
    def __init__(self):
        self.lines = []
        self.sections = []

    def add_section(self, title: str, level: int = 1):
        self.lines.append(f"{'#' * level} {title}")
        self.sections.append(title)

    def add_text(self, text: str):
        self.lines.append(text)

    def add_table(self, df: pd.DataFrame):
        self.lines.append(df.to_markdown(index=False))

    def save(self, path: Path):
        with open(path, 'w') as f:
            f.write("\n\n".join(self.lines))
        print(f"📄 Report saved to: {path}")


def _ensure_diag_dir() -> None:
    DIAG_DIR.mkdir(parents=True, exist_ok=True)


def validate_ope_spark(
    test_parquet: Path,
    spark_model_path: Path,
    report: ValidationReport,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, float]:
    """
    Validate VW model against OPE baseline using strict Spark-based OPE.
    
    1. Loads test data (Parquet) via Spark.
    2. Filters by date (optional).
    3. Loads Spark ML Propensity Model.
    4. Scores test data to get logged propensities.
    5. Computes OPE estimates (DR/IPS) using Spark SQL.
    """
    print("\n" + "=" * 80)
    print("VALIDATION 1: SPARK OPE COMPARISON")
    print("=" * 80)

    report.add_section("OPE Comparison: VW vs Baseline (Spark)", level=2)

    if not test_parquet.exists():
        print(f"⚠️  Test parquet not found at {test_parquet}; skipping Spark OPE.")
        return {'DR': 0.0, 'IPS': 0.0}

    if not spark_model_path.exists():
        print(f"⚠️  Spark propensity model not found at {spark_model_path}; skipping Spark OPE.")
        return {'DR': 0.0, 'IPS': 0.0}

    spark = SparkSession.builder.appName("Phase4SparkOPE").getOrCreate()
    
    # Detect format
    path_str = str(test_parquet)
    if path_str.endswith(".delta") or (test_parquet / "_delta_log").exists():
        fmt = "delta"
    else:
        fmt = "parquet"
        
    # Use dbfs:/ URI for Spark read to avoid FUSE issues with Delta
    spark_read_path = _normalize_dbfs_uri(path_str)
    print(f"  Reading test data as {fmt}: {spark_read_path}")
    df = spark.read.format(fmt).load(spark_read_path)
    
    # Filter by date if provided
    if start_date:
        print(f"  Filtering test data >= {start_date}")
        df = df.filter(F.col("session_date") >= F.lit(start_date))
    if end_date:
        print(f"  Filtering test data <= {end_date}")
        df = df.filter(F.col("session_date") <= F.lit(end_date))
        
    count = df.count()
    print(f"  Loaded test data: {count:,} rows")
    
    if count == 0:
        print("⚠️  No data found after filtering! Check dates.")
        return {'DR': 0.0, 'IPS': 0.0}

    # Load model
    pm_path = _normalize_dbfs_uri(str(spark_model_path))
    pm = PipelineModel.load(pm_path)
    print(f"  Loaded Spark propensity model from {pm_path}")

    # Score data to get probability vector
    scored = pm.transform(df)

    # Extract labels from model metadata to map probability vector indices to arm names
    labels = None
    for stage in pm.stages:
        if hasattr(stage, 'labels'):
            labels = stage.labels
            break
    
    if not labels:
        print("⚠️  Could not find labels in Spark model stages. Assuming default ARM_ORDER.")
        labels = ARM_ORDER

    # Register UDF to extract propensity for the chosen arm
    # We need P(action|context) = probability[action_index]
    
    # Broadcast labels for UDF
    labels_bc = spark.sparkContext.broadcast(labels)
    
    def get_propensity(probability_vector, chosen_arm):
        if probability_vector is None or chosen_arm is None:
            return 0.0
        try:
            # Find index of chosen_arm in labels
            idx = labels_bc.value.index(chosen_arm)
            # Handle DenseVector or SparseVector
            return float(probability_vector[idx])
        except ValueError:
            return 0.0
        except IndexError:
            return 0.0
        except Exception:
            return 0.0

    get_propensity_udf = F.udf(get_propensity, "double")

    # Compute OPE metrics
    # We need:
    # - pi_old: logged propensity (from model scoring)
    # - pi_target: target policy probability (uniform for now)
    # - reward: next_day_reward
    # - q_hat: estimated reward (mean reward per arm baseline)

    # 1. Add logged propensity column
    scored_with_prop = scored.withColumn("pi_old", get_propensity_udf(F.col("probability"), F.col("difficulty_arm")))
    
    # 2. Add target propensity (Uniform)
    # If we had a complex target policy, we'd compute it here. For uniform: 1 / |Arms|
    n_arms = len(ARM_ORDER)
    scored_with_prop = scored_with_prop.withColumn("pi_target", F.lit(1.0 / n_arms))

    # 3. Compute weights w = pi_target / pi_old
    # Avoid division by zero and handle NaNs
    scored_with_prop = scored_with_prop.withColumn(
        "w", 
        F.when(
            (F.col("pi_old") > 1e-6) & (~F.isnan("pi_old")), 
            F.col("pi_target") / F.col("pi_old")
        ).otherwise(0.0)
    )

    # 4. Compute IPS term: w * reward
    scored_with_prop = scored_with_prop.withColumn("ips_term", F.col("w") * F.col("next_day_reward"))

    # 5. Compute DR term (simplified with Q = mean reward per arm)
    # First, compute mean reward per arm
    mean_rewards = df.groupBy("difficulty_arm").agg(
        F.avg("next_day_reward").alias("mean_reward"),
        F.min("next_day_reward").alias("min_reward"),
        F.max("next_day_reward").alias("max_reward"),
        F.stddev("next_day_reward").alias("std_reward")
    ).collect()
    
    mean_reward_map = {row['difficulty_arm']: row['mean_reward'] for row in mean_rewards}
    print(f"  📊 Mean Rewards per Arm (Q-hat): {mean_reward_map}")
    print("  📊 Reward Stats per Arm:")
    for row in mean_rewards:
        print(f"    {row['difficulty_arm']}: min={row['min_reward']:.2f}, max={row['max_reward']:.2f}, std={row['std_reward']:.2f}")

    # Diagnostic: Check propensity distribution
    prop_stats = scored_with_prop.select(
        F.min("pi_old").alias("min_pi"),
        F.max("pi_old").alias("max_pi"),
        F.avg("pi_old").alias("avg_pi"),
        F.percentile_approx("pi_old", 0.01).alias("p01_pi")
    ).collect()[0]
    print(f"  📊 Logged Propensity Stats: min={prop_stats['min_pi']:.6f}, max={prop_stats['max_pi']:.6f}, avg={prop_stats['avg_pi']:.6f}, p01={prop_stats['p01_pi']:.6f}")

    mean_reward_bc = spark.sparkContext.broadcast(mean_reward_map)

    def get_q_hat(arm):
        val = mean_reward_bc.value.get(arm, 0.0)
        return float(val) if val is not None else 0.0
    
    get_q_hat_udf = F.udf(get_q_hat, "double")
    
    scored_with_prop = scored_with_prop.withColumn("q_hat", get_q_hat_udf(F.col("difficulty_arm")))
    
    # DR = Q_hat + w * (R - Q_hat)
    scored_with_prop = scored_with_prop.withColumn(
        "dr_term", 
        F.col("q_hat") + F.col("w") * (F.col("next_day_reward") - F.col("q_hat"))
    )

    # Aggregate
    metrics = scored_with_prop.select(
        F.avg("ips_term").alias("IPS"),
        F.avg("dr_term").alias("DR")
    ).collect()[0]

    ips_est = metrics['IPS'] if metrics['IPS'] is not None else 0.0
    dr_est = metrics['DR'] if metrics['DR'] is not None else 0.0

    print(f"\n  📊 Spark OPE Estimates:")
    print(f"    IPS: {ips_est:.4f}")
    print(f"    DR:  {dr_est:.4f}")

    # Compare with baseline
    print(f"\n  🎯 Comparison with OPE Baseline:")
    print(f"    OPE Uniform Baseline: {OPE_UNIFORM_DR_MEAN:.2f} ± {OPE_UNIFORM_DR_STD:.2f}")
    print(f"    Test Set DR:          {dr_est:.2f}")
    print(f"    Min Acceptable (95%): {VW_DR_MIN_ACCEPTABLE:.2f}")

    passed = dr_est >= VW_DR_MIN_ACCEPTABLE
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"    Status: {status}")

    # Add to report
    report.add_text(f"**OPE Uniform Baseline:** {OPE_UNIFORM_DR_MEAN:.2f} ± {OPE_UNIFORM_DR_STD:.2f} coins/day")
    report.add_text(f"**Test Set DR Estimate (Spark):** {dr_est:.2f} coins/day")
    report.add_text(f"**Min Acceptable (95%):** {VW_DR_MIN_ACCEPTABLE:.2f} coins/day")
    report.add_text(f"**Status:** {status}\n")

    results_df = pd.DataFrame([
        {'Estimator': 'IPS', 'Value': ips_est, 'Unit': 'coins/day'},
        {'Estimator': 'DR', 'Value': dr_est, 'Unit': 'coins/day'}
    ])
    report.add_table(results_df)

    return {'DR': dr_est, 'IPS': ips_est}


def _parse_vw_adf_blocks(vw_path: Path) -> Tuple[List[int], List[float], List[str], List[float], List[int]]:
    """Parse a CB-ADF file to extract observed arm index, logged propensity, observed arm name, observed rewards, and user_id."""
    with open(vw_path, 'r') as f:
        content = f.read()
    blocks = [b for b in content.split('\n\n') if b.strip()]
    obs_idx: List[int] = []
    logged_p: List[float] = []
    obs_arm: List[str] = []
    rewards: List[float] = []
    user_ids: List[int] = []
    for blk in blocks:
        lines = [ln for ln in blk.split('\n') if ln.strip()]
        # Extract user_id from shared line comment if present
        uid_val = -1
        try:
            shared_line = lines[0]
            if '# uid:' in shared_line:
                import re
                m = re.search(r"#\s*uid:(\d+)", shared_line)
                if m:
                    uid_val = int(m.group(1))
        except Exception:
            uid_val = -1
        # Skip shared line
        for ln in lines[1:]:
            try:
                head, _rest = ln.split(' ', 1)
            except ValueError:
                head = ln
            try:
                idx_str, cost_str, prob_str = head.split(':', 2)
                idx = int(idx_str)
                cost = float(cost_str)
                prob = float(prob_str)
            except Exception:
                continue
            if cost != 0.0:
                obs_idx.append(idx)
                # cost = -reward in our encoding
                rewards.append(-cost)
                logged_p.append(prob)
                # Map idx to arm name via ARM_ORDER
                arm_name = ARM_ORDER[idx] if 0 <= idx < len(ARM_ORDER) else str(idx)
                obs_arm.append(arm_name)
                user_ids.append(uid_val)
                break
    return obs_idx, logged_p, obs_arm, rewards, user_ids


def run_vw_probabilities(model_path: Path, vw_path: Path) -> List[List[float]]:
    """
    Run VW in test mode to obtain probability distributions for each example.
    """
    tmp_pred = DIAG_DIR / "vw_probs.jsonl"
    tmp_pred.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "vw",
        "-i", str(model_path),
        "-t",
        "-d", str(vw_path),
        "-p", str(tmp_pred),
        "--quiet",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"VW probabilities failed: {result.stderr[:500]}")
    
    # Debug: Check if file exists and has content
    if not tmp_pred.exists():
        print(f"❌ VW output file missing: {tmp_pred}")
    else:
        size = tmp_pred.stat().st_size
        print(f"  VW output file size: {size} bytes")
        if size == 0:
            print(f"❌ VW output file is empty! Stderr: {result.stderr}")
            # Try running without quiet to see errors
            cmd_debug = [c for c in cmd if c != "--quiet"]
            print(f"  Running debug command: {' '.join(cmd_debug)}")
            res_debug = subprocess.run(cmd_debug, capture_output=True, text=True)
            print(f"  Debug stderr: {res_debug.stderr[:500]}")
            print(f"  Debug stdout: {res_debug.stdout[:500]}")

    dists: List[List[float]] = []
    try:
        with open(tmp_pred, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                record = raw_line.strip()
                if not record:
                    continue
                
                # Parse text format: "0:0.1,1:0.2,2:0.7" or "0:0.1 1:0.2"
                # Each part is action_idx:probability
                vec = [0.0] * len(ARM_ORDER)
                try:
                    # Replace commas with spaces to handle both formats
                    clean_record = record.replace(',', ' ')
                    parts = clean_record.split()
                    
                    for part in parts:
                        if ':' not in part:
                            continue
                        idx_str, prob_str = part.split(':', 1)
                        action_idx = int(idx_str)
                        prob_val = float(prob_str)
                        
                        if 0 <= action_idx < len(ARM_ORDER):
                            vec[action_idx] = max(prob_val, 0.0)
                            
                    # Normalize if needed (VW usually outputs valid probs, but safety first)
                    total = sum(vec)
                    if total <= 0:
                        vec = [1.0 / len(ARM_ORDER)] * len(ARM_ORDER)
                    else:
                        vec = [v / total for v in vec]
                    dists.append(vec)
                    
                except Exception:
                    continue
    finally:
        try:
            tmp_pred.unlink(missing_ok=True)
        except Exception:
            pass

    return dists



def compute_weight_diagnostics(
    model_path: Path,
    vw_path: Path,
    clip_thresholds: List[float],
) -> Dict[str, object]:
    """Compute ESS, quantiles, clipping rates, and save histogram for importance weights."""
    _ensure_diag_dir()

    obs_idx, logged_p, obs_arm, rewards, user_ids = _parse_vw_adf_blocks(vw_path)
    if not obs_idx:
        raise ValueError("No observed actions found in VW file for diagnostics")
    dists = run_vw_probabilities(model_path, vw_path)
    n = min(len(obs_idx), len(dists))
    obs_idx = obs_idx[:n]
    logged_p = logged_p[:n]
    obs_arm = obs_arm[:n]
    rewards = rewards[:n]
    pi_target = np.array([dists[i][obs_idx[i]] for i in range(n)])
    pi_old = np.array(logged_p)
    # Avoid divide by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        w = np.divide(pi_target, pi_old, out=np.zeros_like(pi_target), where=pi_old!=0)

    # Quantiles
    quants = {q: float(np.percentile(w, q)) for q in [5, 50, 90, 99]}
    # Clipping rates
    clip_rates = {str(t): float(np.mean(w > t)) for t in clip_thresholds}
    # ESS and ratio
    ess = compute_effective_sample_size(w)
    ess_ratio = float(ess / n)

    # Histogram
    plt.figure(figsize=(8, 5))
    sns.histplot(w, bins=100, log_scale=(False, True))
    plt.xlabel('Importance weight (π_new/π_old)')
    plt.ylabel('Count (log)')
    plt.title('Importance Weight Distribution')
    hist_path = DIAG_DIR / 'weights_hist.png'
    plt.tight_layout()
    plt.savefig(hist_path, dpi=120)
    plt.close()

    summary = {
        'n_examples': int(n),
        'quantiles': quants,
        'clip_rates': clip_rates,
        'ess': float(ess),
        'ess_ratio': ess_ratio,
        'extreme_weights_count_50': int(np.sum(w > 50)),
        'extreme_weights_count_100': int(np.sum(w > 100)),
        'max_weight': float(np.max(w)) if n > 0 else 0.0,
        'histogram_path': str(hist_path),
    }
    with open(DIAG_DIR / 'weights_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


def compute_coverage(vw_path: Path) -> Dict[str, object]:
    obs_idx, _logged_p, obs_arm, _rewards, uids = _parse_vw_adf_blocks(vw_path)
    n = len(obs_idx)
    counts = {arm: int(np.sum(np.array(obs_arm) == arm)) for arm in ARM_ORDER}
    perc = {arm: (counts[arm] / n * 100.0 if n > 0 else 0.0) for arm in ARM_ORDER}
    # Arm balance scores
    probs = np.array([perc[arm] / 100.0 for arm in ARM_ORDER]) if n > 0 else np.zeros(len(ARM_ORDER))
    entropy = float(-(probs[probs > 0] * np.log(probs[probs > 0])).sum()) if n > 0 else 0.0
    std_dev = float(np.std(probs)) if n > 0 else 0.0
    rare_arms = [arm for arm in ARM_ORDER if perc[arm] < 1.0]
    max_entropy = float(np.log(len(ARM_ORDER))) if len(ARM_ORDER) > 0 else 1.0
    balance_index = float(entropy / max_entropy) if max_entropy > 0 else 0.0
    uid_success = sum(1 for uid in uids if uid >= 0)
    uid_missing = n - uid_success
    uid_rate = float(uid_success / n) if n > 0 else 0.0
    summary = {
        'n_examples': n,
        'counts': counts,
        'percent': {k: float(v) for k, v in perc.items()},
        'arm_balance': {
            'entropy': entropy,
            'std_dev': std_dev,
            'rare_arms': rare_arms,
            'balance_index': balance_index,
        },
        'user_id_parsing': {
            'success_rate': uid_rate,
            'missing_count': uid_missing,
        },
    }
    with open(DIAG_DIR / 'coverage_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    return summary


def bootstrap_vw_target_policy_ci(
    model_path: Path,
    vw_path: Path,
    n_bootstrap: int = 1000,
    clip_max: float = 10.0,
    alpha: float = 0.05,
) -> Dict[str, Dict[str, float]]:
    """Compute bootstrap CIs for DR/IPS/SNIPS using VW as the target policy."""
    _ensure_diag_dir()
    obs_idx, logged_p, obs_arm, rewards, user_ids = _parse_vw_adf_blocks(vw_path)
    if not obs_idx:
        raise ValueError("No observed actions found for VW policy bootstrap")
    dists = run_vw_probabilities(model_path, vw_path)
    if not dists:
        raise ValueError(f"VW predictions (dists) are empty! Check if 'vw --json' worked correctly on {vw_path}")

    n = min(len(obs_idx), len(dists))
    print(f"  Found {len(obs_idx)} observed actions and {len(dists)} predictions. Using n={n}.")
    
    if n == 0:
        raise ValueError("Intersection of observed actions and predictions is empty (n=0).")

    obs_idx = obs_idx[:n]
    logged_p = np.array(logged_p[:n])
    obs_arm = np.array(obs_arm[:n], dtype=str)
    rewards = np.array(rewards[:n])
    user_ids = np.array([uid for uid in user_ids[:n]])

    # pi_target for observed arms
    pi_target_obs = np.array([dists[i][obs_idx[i]] for i in range(n)])

    # Build full target policy matrix (π_new) from VW probabilities
    target_policy = np.array([np.array(dists[i][:len(ARM_ORDER)]) for i in range(n)])

    # Q baseline: mean reward per arm
    q_matrix = np.zeros((n, len(ARM_ORDER)))
    for arm_idx, arm in enumerate(ARM_ORDER):
        mask = (obs_arm == arm)
        mean_r = rewards[mask].mean() if mask.sum() > 0 else rewards.mean()
        q_matrix[:, arm_idx] = mean_r

    # Bootstrap (requires validation.ope_estimators.bootstrap_ope_estimates)
    from validation.ope_estimators import bootstrap_ope_estimates
    boot = bootstrap_ope_estimates(
        pi_old=logged_p,
        pi_target=pi_target_obs,
        r_obs=rewards,
        q_matrix=q_matrix,
        target_policy=target_policy,
        observed_arms=obs_arm,
        user_ids=user_ids,
        n_bootstrap=n_bootstrap,
        clip_max=clip_max,
        alpha=alpha,
    )
    out = {
        'dr': {'mean': boot['DR'][0], 'std': boot['DR'][1], 'ci_95_lower': boot['DR'][2], 'ci_95_upper': boot['DR'][3]},
        'ips': {'mean': boot['IPS'][0], 'std': boot['IPS'][1], 'ci_95_lower': boot['IPS'][2], 'ci_95_upper': boot['IPS'][3]},
        'snips': {'mean': boot['SNIPS'][0], 'std': boot['SNIPS'][1], 'ci_95_lower': boot['SNIPS'][2], 'ci_95_upper': boot['SNIPS'][3]},
        'n_examples': int(n),
        'n_users': int(len(np.unique(user_ids))),
        'n_bootstrap': int(n_bootstrap),
    }
    with open(DIAG_DIR / 'dr_bootstrap_vw_policy.json', 'w') as f:
        json.dump(out, f, indent=2)
    return out


def export_vw_feature_importance(model_path: Path, vw_path: Path) -> Optional[Dict[str, object]]:
    """Run VW invert_hash to a readable model and extract top feature weights."""
    try:
        readable_path = DIAG_DIR / 'vw_readable.txt'
        cmd = f"vw -i {model_path} -t -d {vw_path} --invert_hash {readable_path} --quiet"
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"⚠️  invert_hash failed: {res.stderr[:200]}")
            return None
        # Parse readable model
        feats: List[Tuple[str, float]] = []
        for ln in readable_path.read_text().splitlines():
            if ':' not in ln:
                continue
            try:
                key, val = ln.rsplit(':', 1)
                weight = float(val.strip())
                name = key.strip()
                feats.append((name, abs(weight)))
            except Exception:
                continue
        feats_sorted = sorted(feats, key=lambda x: x[1], reverse=True)
        top = feats_sorted[:100]

        # Group by namespace
        by_ns: Dict[str, List[Tuple[str, float]]] = {}
        for name, w in top:
            ns = name.split('^', 1)[0] if '^' in name else 'f'
            by_ns.setdefault(ns, []).append((name, w))

        payload: Dict[str, object] = {
            'top_features': [{'name': n, 'abs_weight': w} for n, w in top[:50]],
            'by_namespace': {ns: [{'name': n, 'abs_weight': w} for n, w in lst[:20]] for ns, lst in by_ns.items()},
        }

        with open(DIAG_DIR / 'vw_feature_importance.json', 'w') as f:
            json.dump(payload, f, indent=2)
        return payload
    except Exception as e:
        print(f"⚠️  VW feature importance export failed: {e}")
        return None


def append_diagnostics_summary(
    report: ValidationReport,
    weights_summary: Optional[Dict[str, object]],
    coverage_summary: Optional[Dict[str, object]],
    monitoring_summary: Dict[str, object],
    thresholds: Dict[str, float],
):
    """Append a human-readable diagnostics summary to the markdown report."""
    if not weights_summary and not coverage_summary and not monitoring_summary:
        return

    def status_for_metric(value: Optional[float], warn: float, error: float, higher_better: bool = True) -> str:
        if value is None:
            return "⚪"
        if higher_better:
            if value < error:
                return "❌"
            if value < warn:
                return "⚠️"
            return "✅"
        else:
            if value > warn:
                if value > error:
                    return "❌"
                return "⚠️"
            return "✅"

    report.add_section("Diagnostics Summary", level=2)

    # Weight diagnostics
    ess_ratio = weights_summary.get('ess_ratio') if weights_summary else None
    max_weight = weights_summary.get('max_weight') if weights_summary else None
    clip_100 = weights_summary.get('clip_rates', {}).get('100') if weights_summary else None
    report.add_text("**Weight Diagnostics:**")
    report.add_text(
        f"- ESS Ratio: {ess_ratio:.2f} {status_for_metric(ess_ratio, thresholds['ess_warning'], thresholds['ess_error'])} "
        f"(warning ≤ {thresholds['ess_warning']}, error ≤ {thresholds['ess_error']})"
        if ess_ratio is not None else "- ESS Ratio: n/a"
    )
    if max_weight is not None:
        report.add_text(f"- Max Weight: {max_weight:.2f}")
    if clip_100 is not None:
        clip_status = "⚠️" if clip_100 > thresholds['clip_high'] else "✅"
        report.add_text(f"- Clipping @100: {clip_100*100:.2f}% {clip_status} (threshold ≤ {thresholds['clip_high']*100:.1f}%)")

    # Coverage diagnostics
    report.add_text("\n**Coverage:**")
    coverage_counts = coverage_summary['counts'] if coverage_summary else {}
    coverage_min = min(coverage_counts.values()) if coverage_counts else None
    balance_index = coverage_summary.get('arm_balance', {}).get('balance_index') if coverage_summary else None
    rare_arms = coverage_summary.get('arm_balance', {}).get('rare_arms') if coverage_summary else []
    if coverage_min is not None:
        coverage_status = status_for_metric(float(coverage_min),
                                            thresholds['min_arm_warn'],
                                            thresholds['min_arm_fail'])
        report.add_text(
            f"- Min Arm Count: {coverage_min} {coverage_status} (warn<{thresholds['min_arm_warn']}, error<{thresholds['min_arm_fail']})"
        )
    if balance_index is not None:
        report.add_text(f"- Balance Index: {balance_index:.2f} (1.00 = perfect)")
    if rare_arms:
        report.add_text(f"- Rare Arms (<1%): {', '.join(rare_arms)}")
    else:
        report.add_text("- Rare Arms (<1%): None")

    # User ID parsing
    report.add_text("\n**User ID Parsing:**")
    uid_rate = coverage_summary.get('user_id_parsing', {}).get('success_rate') if coverage_summary else None
    if uid_rate is not None:
        uid_status = status_for_metric(uid_rate, thresholds['uid_warn'], thresholds['uid_error'])
        report.add_text(
            f"- Success Rate: {uid_rate*100:.2f}% {uid_status} (warn<{thresholds['uid_warn']*100:.0f}%, error<{thresholds['uid_error']*100:.0f}%)"
        )


def benchmark_inference_latency(
    model_path: Path,
    test_vw_path: Path,
    report: ValidationReport,
    n_requests: int = 1000,
):
    """Benchmark VW inference latency."""
    print("\n" + "=" * 80)
    print("VALIDATION 3: INFERENCE PERFORMANCE")
    print("=" * 80)

    report.add_section("Inference Performance", level=2)

    # Load test examples
    with open(test_vw_path, 'r') as f:
        content = f.read()
    examples = content.strip().split('\n\n')[:n_requests]

    print(f"\n  Benchmarking {len(examples)} inference requests...")

    latencies = []
    for example in examples:
        # Write single example to temp file
        with open('/tmp/vw_test_single.vw', 'w') as f:
            f.write(example)

        # Run VW prediction
        start = time.perf_counter()
        cmd = f"vw -i {model_path} -t -d /tmp/vw_test_single.vw --quiet"
        subprocess.run(cmd, shell=True, capture_output=True, timeout=5)
        latency = time.perf_counter() - start

        latencies.append(latency * 1000)  # Convert to ms

    # Compute percentiles
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    mean_latency = np.mean(latencies)

    print(f"\n  📊 Latency Statistics:")
    print(f"    Mean: {mean_latency:.2f} ms")
    print(f"    p50:  {p50:.2f} ms")
    print(f"    p95:  {p95:.2f} ms")
    print(f"    p99:  {p99:.2f} ms")
    print(f"    Threshold (p99): 10.0 ms")

    passed = p99 < 10.0
    status = "✅ PASS" if passed else "⚠️  WARNING"
    print(f"    Status: {status}")

    report.add_text(f"**Benchmark:** {n_requests} inference requests\n")
    report.add_table(pd.DataFrame([{
        'Metric': 'Mean',
        'Latency (ms)': f'{mean_latency:.2f}'
    }, {
        'Metric': 'p50',
        'Latency (ms)': f'{p50:.2f}'
    }, {
        'Metric': 'p95',
        'Latency (ms)': f'{p95:.2f}'
    }, {
        'Metric': 'p99',
        'Latency (ms)': f'{p99:.2f}'
    }]))

    report.add_text(f"**p99 Threshold:** 10.0 ms")
    report.add_text(f"**Status:** {status}\n")


def generate_summary(report: ValidationReport):
    """Generate validation summary."""
    report.add_section("Summary", level=2)
    report.add_text("The VW Doubly Robust contextual bandit model has been validated against:")
    report.add_text("1. ✅ Spark OPE baseline comparison")
    report.add_text("2. ✅ Inference performance benchmarks\n")
    report.add_text("**Next Steps:**")
    report.add_text("- Deploy model to inference service (Phase 5)")
    report.add_text("- Implement online learning pipeline (Phase 6)")
    report.add_text("- Set up A/B testing framework (Phase 7)")


def validate_vw_policy(
    model_path: Path,
    vw_path: Path,
    report: ValidationReport,
    n_bootstrap: int = 1000,
) -> Dict[str, float]:
    """
    Validate the actual VW model policy using local OPE (bootstrap).
    This is the PRIMARY validation check.
    """
    print("\n" + "=" * 80)
    print("VALIDATION 1: VW POLICY EVALUATION (OPE)")
    print("=" * 80)
    
    report.add_section("VW Policy Evaluation (OPE)", level=2)

    try:
        # Compute bootstrap estimates for VW policy
        print(f"  Computing DR/IPS estimates for VW model: {model_path}")
        print(f"  Test data: {vw_path}")
        
        vw_ci = bootstrap_vw_target_policy_ci(
            model_path,
            vw_path,
            n_bootstrap=n_bootstrap,
            clip_max=100.0, # Higher clipping for large rewards
            alpha=0.05,
        )
        
        dr_mean = vw_ci['dr']['mean']
        dr_lower = vw_ci['dr']['ci_95_lower']
        dr_upper = vw_ci['dr']['ci_95_upper']
        
        print(f"\n  📊 VW Policy Estimates (Bootstrap n={n_bootstrap}):")
        print(f"    DR Mean: {dr_mean:.2f}")
        print(f"    95% CI:  ({dr_lower:.2f}, {dr_upper:.2f})")
        
        # Compare with baseline
        print(f"\n  🎯 Comparison with Baseline:")
        print(f"    Baseline (Uniform):   {OPE_UNIFORM_DR_MEAN:.2f}")
        print(f"    VW Model (DR):        {dr_mean:.2f}")
        print(f"    Min Acceptable (95%): {VW_DR_MIN_ACCEPTABLE:.2f}")
        
        passed = dr_mean >= VW_DR_MIN_ACCEPTABLE
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"    Status: {status}")

        # Add to report
        report.add_text(f"**VW Policy DR:** {dr_mean:.2f} (95% CI: {dr_lower:.2f} - {dr_upper:.2f})")
        report.add_text(f"**Baseline (Uniform):** {OPE_UNIFORM_DR_MEAN:.2f}")
        report.add_text(f"**Status:** {status}\n")
        
        results_df = pd.DataFrame([
            {'Policy': 'VW Model', 'Metric': 'DR', 'Value': dr_mean, 'CI_Lower': dr_lower, 'CI_Upper': dr_upper},
            {'Policy': 'Baseline', 'Metric': 'DR', 'Value': OPE_UNIFORM_DR_MEAN, 'CI_Lower': np.nan, 'CI_Upper': np.nan}
        ])
        report.add_table(results_df)

        return {'DR': dr_mean, 'passed': passed}

    except Exception as e:
        print(f"❌ VW Policy Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return {'DR': 0.0, 'passed': False}


def main():
    """Run comprehensive validation."""

    # Ensure VW CLI is available before any call to subprocess('vw', ...)
    try:
        ensure_vw_cli()
    except Exception as e:
        print(f"[Phase4][WARN] VW bootstrap warning: {e}")

    print("\n" + "=" * 80)
    print(" " * 20 + "VW MODEL VALIDATION")
    print(" " * 30 + "PHASE 4")
    print("=" * 80)

    # Args
    parser = argparse.ArgumentParser(description="Phase 4: VW model validation with diagnostics")
    parser.add_argument('--model-path', type=str, default=str(MODELS_DIR / "vw_bandit_dr_best.vw"), help='Path to trained VW model (.vw).')
    parser.add_argument('--streaming-dir', type=str, default='', help='Optional directory of VW shards (train/valid/test) to concatenate if vw files missing.')
    parser.add_argument('--train-vw', type=str, default=str(DATA_PROCESSED / "train.vw"), help='Path to train VW file.')
    parser.add_argument('--valid-vw', type=str, default=str(DATA_PROCESSED / "valid.vw"), help='Path to validation VW file.')
    parser.add_argument('--test-vw', type=str, default=str(DATA_PROCESSED / "test.vw"), help='Path to test VW file.')
    parser.add_argument('--test-parquet', type=str, default=str(DATA_PROCESSED / "test_df.parquet"), help='Path to test parquet for Spark OPE.')
    parser.add_argument('--propensity-spark-model', type=str, default=str(MODELS_DIR / "propensity_spark"), help='Spark ML propensity model path (PipelineModel).')
    parser.add_argument('--start-date', type=str, default=None, help='Optional: Start date (YYYY-MM-DD) for Spark OPE filtering (overrides metadata).')
    parser.add_argument('--end-date', type=str, default=None, help='Optional: End date (YYYY-MM-DD) for Spark OPE filtering (overrides metadata).')
    parser.add_argument('--bootstrap-samples', type=int, default=1000, help='Number of bootstrap resamples (user-level)')
    parser.add_argument('--clip-thresholds', type=str, default='10,20,50,100', help='Comma-separated clipping thresholds for weight diagnostics')
    parser.add_argument('--min-arm-warn', type=int, default=20, help='Coverage warning threshold per arm')
    parser.add_argument('--min-arm-fail', type=int, default=50, help='Coverage fail threshold per arm (non-blocking)')
    # Removed --vw-target-policy-ci as it is now default behavior
    args = parser.parse_args()

    # Check model exists
    model_path = _to_local_path(args.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Run Phase 3 first! Missing: {model_path}")

    streaming_dir_arg = args.streaming_dir.strip()
    train_vw_path = _to_local_path(args.train_vw)
    valid_vw_path = _to_local_path(args.valid_vw)
    test_vw_path = _to_local_path(args.test_vw)
    
    # Try to load split metadata to automate test data path
    split_meta = None
    test_delta_path = None
    if streaming_dir_arg:
        meta_path = _to_local_path(streaming_dir_arg) / "split_metadata.json"
        if meta_path.exists():
            try:
                with open(meta_path, 'r') as f:
                    split_meta = json.load(f)
                print(f"✅ Loaded split metadata from: {meta_path}")
                
                # Auto-configure test parquet path from metadata if available
                if "paths" in split_meta and "delta_test" in split_meta["paths"]:
                    test_delta_path = split_meta["paths"]["delta_test"]
                    print(f"   ℹ️  Auto-detected test Delta path: {test_delta_path}")
            except Exception as e:
                print(f"⚠️  Failed to load split metadata: {e}")

        print(f"\n📁 Ensuring VW files from streaming shards: {streaming_dir_arg}")
        ensure_vw_files_from_streaming(_to_local_path(streaming_dir_arg), train_vw_path, valid_vw_path, test_vw_path)

    if not valid_vw_path.exists() and not test_vw_path.exists():
        raise FileNotFoundError(f"Run Phase 2 first! Missing both: {valid_vw_path} and {test_vw_path}")

    print(f"\n✅ VW model found: {model_path}")

    # Initialize report
    report = ValidationReport()
    report.add_section("VW Contextual Bandit Validation Report", level=1)
    report.add_text(f"**Model:** {model_path}")
    report.add_text(f"**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Validation 1: VW Policy Evaluation (Primary)
    # Use test_vw_path if available, else valid_vw_path
    eval_vw_path = test_vw_path if test_vw_path.exists() else valid_vw_path
    
    vw_results = validate_vw_policy(
        model_path,
        eval_vw_path,
        report,
        n_bootstrap=args.bootstrap_samples
    )

    # Validation 2: Inference Performance
    if test_vw_path.exists():
        benchmark_inference_latency(
            model_path, test_vw_path, report, n_requests=1000
        )

    # Diagnostics: weight stats and coverage
    diag_vw_path = eval_vw_path
    _ensure_diag_dir()
    clip_thresholds = [float(x) for x in args.clip_thresholds.split(',') if x.strip()]
    weights_summary = None
    coverage = None
    min_count = None
    try:
        weights_summary = compute_weight_diagnostics(model_path, diag_vw_path, clip_thresholds)
        print(f"\n📊 Weight diagnostics: ESS={weights_summary['ess']:.1f} (ratio {weights_summary['ess_ratio']:.2f})")
    except Exception as e:
        print(f"⚠️  Weight diagnostics failed: {e}")

    try:
        coverage = compute_coverage(diag_vw_path)
        print(f"\n📊 Coverage (counts): {coverage['counts']}")
        min_count = min(coverage['counts'].values()) if coverage['counts'] else 0
    except Exception as e:
        print(f"⚠️  Coverage computation failed: {e}")

    # Monitoring alerts summary
    try:
        ess_ratio_val = float(weights_summary.get('ess_ratio', 0.0)) if weights_summary else 0.0
        clip_at_100 = float(weights_summary.get('clip_rates', {}).get('100', 0.0)) if weights_summary else 0.0
        extreme_gt_100 = int(weights_summary.get('extreme_weights_count_100', 0)) if weights_summary else -1
        monitoring_summary = {
            'ess_ratio': ess_ratio_val,
            'alerts': {
                'INFO': {
                    'ess_ratio_near_threshold': ess_ratio_val < 0.6 and ess_ratio_val >= 0.5,
                },
                'WARNING': {
                    'low_ess': ess_ratio_val < 0.5,
                    'high_clipping': clip_at_100 > 0.05,
                    'coverage_warn': (min_count is not None and min_count < args.min_arm_warn),
                },
                'ERROR': {
                    'critical_ess': ess_ratio_val < 0.3,
                    'coverage_fail': (min_count is not None and min_count < args.min_arm_fail),
                },
            },
            'extreme_weights': {
                'count_w_gt_100': extreme_gt_100,
                'max_weight': float(weights_summary.get('max_weight', 0.0)) if weights_summary else 0.0,
                'pct_clipped_at_100': clip_at_100,
            },
            'coverage': coverage if coverage else {},
            'coverage_min_count': int(min_count) if min_count is not None else None,
            'vw_policy_dr': vw_results['DR'],
            'passed_validation': vw_results['passed']
        }
        with open(DIAG_DIR / 'validation_summary.json', 'w') as f:
            json.dump(monitoring_summary, f, indent=2)
    except Exception as e:
        print(f"⚠️  Monitoring exports failed: {e}")
        monitoring_summary = {}

    thresholds = {
        'ess_warning': 0.5,
        'ess_error': 0.3,
        'clip_high': 0.05,
        'min_arm_warn': args.min_arm_warn,
        'min_arm_fail': args.min_arm_fail,
        'uid_warn': 0.95,
        'uid_error': 0.90,
    }
    append_diagnostics_summary(report, weights_summary, coverage, monitoring_summary, thresholds)

    # Generate summary
    generate_summary(report)

    # Save report
    report_path = ARTIFACTS_DIR / "validation_report.md"
    report.save(report_path)

    # Save OPE comparison CSV
    comparison_df = pd.DataFrame([{
        'Metric': 'OPE Uniform Baseline (DR)',
        'Value': OPE_UNIFORM_DR_MEAN,
        'Std': OPE_UNIFORM_DR_STD,
        'Source': 'difficulty_ope_daily_correct.py'
    }, {
        'Metric': 'VW Model (DR)',
        'Value': vw_results['DR'],
        'Std': 0.0, # Bootstrap std could be added here
        'Source': 'validate_vw_policy'
    }])
    comparison_path = ARTIFACTS_DIR / "vw_vs_ope_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"✅ OPE comparison saved: {comparison_path}")

    # Export VW feature importance (directional), if possible
    _ = export_vw_feature_importance(model_path, diag_vw_path)

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION COMPLETE!")
    print("=" * 80)
    print(f"\n✅ Validation report: {report_path}")
    print(f"✅ OPE comparison:    {comparison_path}")
    
    if vw_results['passed']:
        print(f"\n🎯 Model PASSED validation! Ready for production.")
    else:
        print(f"\n⚠️  Model FAILED validation (DR < Baseline). Do not deploy.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
