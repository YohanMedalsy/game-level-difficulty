#!/usr/bin/env python3
"""
Phase 3: VW Contextual Bandit Training with Optuna Hyperparameter Optimization

Trains Vowpal Wabbit DR (Doubly Robust) contextual bandit using Optuna for hyperparameter search.

Optimizes:
- CB algorithm: dr (fixed)
- Exploration: epsilon-greedy vs epsilon+bagging
- Learning rate, L2 regularization, power_t (decay)
- Feature interactions (quadratic, cubic, specific namespaces)
- Bagging ensemble size

Objective:
1. Minimize VW progressive validation loss (surrogate for maximizing next-day reward)

Outputs:
- models/vw_bandit_dr_best.vw (best model)
- config/vw_config_best.yaml (best hyperparameters)
- artifacts/optuna_study.db (full study database)
- artifacts/optuna_optimization_history.html (interactive plots)
"""

import argparse
import os
import shutil
import sys
import subprocess
import time
import tempfile
from pathlib import Path
import json
import yaml
import re
from typing import Dict, Any, Optional, Tuple, List
from collections import Counter

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler

# Configure logging
import logging

# Log locally first to avoid DBFS "Operation not supported" (Errno 95) during flush
LOCAL_PIPELINE_LOG = "/tmp/training_pipeline.log"
REMOTE_PIPELINE_LOG = "/dbfs/mnt/bandit/logs/training_pipeline.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOCAL_PIPELINE_LOG),
        logging.StreamHandler()
    ]
)
from optuna.visualization import (
    plot_optimization_history,
    plot_param_importances,
)

try:
    import mlflow
    from mlflow import pyfunc

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None
    pyfunc = None


# Add parent directory to path
try:
    CURRENT_FILE = Path(__file__).resolve()
except NameError:
    import inspect
    frame = inspect.currentframe()
    CURRENT_FILE = Path(inspect.getfile(frame)).resolve() if frame else Path.cwd()

BANDIT_ROOT = CURRENT_FILE.parent.parent
sys.path.insert(0, str(BANDIT_ROOT / "src"))

from constants import OPE_UNIFORM_DR_MEAN, ARM_ORDER


# Set default MLflow experiment/model name (can be overridden via env vars)
if "MLFLOW_EXPERIMENT_PATH" not in os.environ:
    os.environ["MLFLOW_EXPERIMENT_PATH"] = "/Users/yohan.medalsy@spaceplay.games/vw_bandit"
if "MLFLOW_MODEL_NAME" not in os.environ:
    os.environ["MLFLOW_MODEL_NAME"] = "vw_bandit_dr"


class VWModelWrapper(pyfunc.PythonModel):
    """
    Thin MLflow pyfunc wrapper around a trained VW .vw model.

    Predict interface:
      - Expects a pandas Series or DataFrame whose first column contains
        VW CB_ADF examples as strings (one or more examples).
      - Returns a pandas Series of raw VW predictions (floats).
    """

    def load_context(self, context):
        # Path to the logged VW model artifact
        self.vw_model_path = context.artifacts["vw_model"]

    def predict(self, context, model_input):
        import pandas as pd

        # Normalize input to a Series of strings
        if isinstance(model_input, pd.DataFrame):
            series = model_input.iloc[:, 0].astype(str)
        else:
            series = pd.Series(model_input).astype(str)

        preds = []
        for example in series:
            # Write single example to a temp file and run vw -p for prediction
            import tempfile
            import subprocess
            import time

            with tempfile.NamedTemporaryFile(mode="w", suffix=".vw", delete=True) as f_in, tempfile.NamedTemporaryFile(
                mode="r", suffix=".txt", delete=True
            ) as f_out:
                f_in.write(example)
                f_in.flush()
                cmd = [
                    "vw",
                    "-i",
                    self.vw_model_path,
                    "-t",
                    "-d",
                    f_in.name,
                    "-p",
                    f_out.name,
                    "--quiet",
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                f_out.seek(0)
                line = f_out.read().strip().split()[0]
                preds.append(float(line))

        return pd.Series(preds)



def _dbfs_to_local(path: str) -> Path:
    """Convert a dbfs:/ URI or local path string into a Path on the driver."""
    if path.startswith("dbfs:/"):
        return Path("/dbfs" + path[5:])
    return Path(path)


# Default DBFS roots (override via env vars to customize persistence locations)
DEFAULT_DBFS_BASE = os.environ.get("VW_DBFS_BASE", "dbfs:/mnt/vw_pipeline")
DEFAULT_TRAIN_VW = os.environ.get("VW_TRAIN_VW", f"{DEFAULT_DBFS_BASE}/data/processed/train.vw")
DEFAULT_VALID_VW = os.environ.get("VW_VALID_VW", f"{DEFAULT_DBFS_BASE}/data/processed/valid.vw")
DEFAULT_TEST_VW = os.environ.get("VW_TEST_VW", f"{DEFAULT_DBFS_BASE}/data/processed/test.vw")
DEFAULT_MODELS_DIR = os.environ.get("VW_MODELS_DIR", f"{DEFAULT_DBFS_BASE}/models")
DEFAULT_CONFIG_DIR = os.environ.get("VW_CONFIG_DIR", f"{DEFAULT_DBFS_BASE}/config")
DEFAULT_ARTIFACTS_DIR = os.environ.get("VW_ARTIFACTS_DIR", f"{DEFAULT_DBFS_BASE}/artifacts")
DEFAULT_LOGS_DIR = os.environ.get("VW_LOGS_DIR", f"{DEFAULT_DBFS_BASE}/logs/training")
DEFAULT_LOCAL_BASE = os.environ.get("VW_LOCAL_BASE", "/tmp/vw_pipeline")
TRAIN_TIMEOUT_SECS = int(os.environ.get("VW_TRAIN_TIMEOUT_SECS", "7200"))

DATA_PROCESSED = _dbfs_to_local(f"{DEFAULT_DBFS_BASE}/data/processed")
MODELS_DIR = _dbfs_to_local(DEFAULT_MODELS_DIR)
ARTIFACTS_DIR = _dbfs_to_local(DEFAULT_ARTIFACTS_DIR)
CONFIG_DIR = _dbfs_to_local(DEFAULT_CONFIG_DIR)
LOGS_DIR = _dbfs_to_local(DEFAULT_LOGS_DIR)

LOCAL_BASE_DIR = Path(DEFAULT_LOCAL_BASE)
if str(LOCAL_BASE_DIR).startswith("dbfs:/"):
    LOCAL_BASE_DIR = _dbfs_to_local(str(LOCAL_BASE_DIR))
LOCAL_ARTIFACTS_DIR = LOCAL_BASE_DIR / "artifacts"
LOCAL_LOGS_DIR = LOCAL_BASE_DIR / "logs" / "training"
LOCAL_MODELS_DIR = LOCAL_BASE_DIR / "models"


def _reset_local_base_if_needed() -> None:
    """
    Recreate the VW scratch directories when targeting ephemeral disks.
    Applies to /local_disk0, /tmp, or /databricks/driver when overridden.
    """
    base = str(LOCAL_BASE_DIR)
    is_ephemeral = base.startswith("/local_disk0") or base.startswith("/tmp") or base.startswith("/databricks/driver")
    if not is_ephemeral:
        return

    current_user = os.environ.get("USER") or "root"
    base_path = str(LOCAL_BASE_DIR)

    try:
        if LOCAL_BASE_DIR.exists():
            subprocess.run(["sudo", "rm", "-rf", base_path], check=False)
        subprocess.run([
            "sudo", "mkdir", "-p",
            f"{base_path}/models",
            f"{base_path}/artifacts",
            f"{base_path}/logs/training",
        ], check=True)
        subprocess.run(["sudo", "chown", "-R", current_user, base_path], check=False)
    except Exception as exc:  # pragma: no cover - best-effort reset
        print(f"[Phase3][WARN] Unable to reset scratch dir {base_path}: {exc}")


_reset_local_base_if_needed()

for directory in (DATA_PROCESSED, MODELS_DIR, ARTIFACTS_DIR, CONFIG_DIR, LOGS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

for directory in (LOCAL_BASE_DIR, LOCAL_ARTIFACTS_DIR, LOCAL_LOGS_DIR, LOCAL_MODELS_DIR):
    directory.mkdir(parents=True, exist_ok=True)

DEBUG_LOG = LOCAL_LOGS_DIR / "phase3_debug.log"
REMOTE_DEBUG_LOG = LOGS_DIR / "phase3_debug.log"
LOCAL_OPTUNA_DB = LOCAL_ARTIFACTS_DIR / "optuna_study.db"
REMOTE_OPTUNA_DB = ARTIFACTS_DIR / "optuna_study.db"

VALID_LOGGED_DATA: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None


def _wait_for_path(path: Path, timeout: float = 120.0, poll: float = 2.0) -> bool:
    """Poll for a path to appear (helps when writing to /dbfs with eventual consistency)."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if path.exists():
            return True
        time.sleep(poll)
    return path.exists()


def _debug_missing_model(path: Path, log_file: Path) -> None:
    try:
        print(f"[Phase3][DEBUG] Missing model at {path}")
        parent = path.parent
        if parent.exists():
            ls = subprocess.run(["ls", "-l", str(parent)], capture_output=True, text=True)
            print(f"[Phase3][DEBUG] Dir listing {parent} ->\n{ls.stdout}")
        if log_file.exists():
            tail = subprocess.run(["tail", "-n", "50", str(log_file)], capture_output=True, text=True)
            print(f"[Phase3][DEBUG] Trial log tail ->\n{tail.stdout}")
    except Exception as exc:
        print(f"[Phase3][WARN] Debug dump failed: {exc}")


def debug(msg: str) -> None:
    """Log diagnostic message to stdout and persistent debug file."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    formatted = f"[{timestamp}][Phase3] {msg}"
    print(formatted, flush=True)
    try:
        with open(DEBUG_LOG, "a", encoding="utf-8") as fh:
            fh.write(formatted + "\n")
    except Exception as exc:  # pragma: no cover
        print(f"[Phase3][WARN] Failed to write debug log: {exc}", flush=True)

def _persist_file(local_path: Path, remote_path: Path, label: str) -> None:
    """Copy a local artifact back to DBFS for persistence."""
    if not local_path.exists():
        debug(f"Skipping persistence for {label}; missing {local_path}")
        return
    try:
        remote_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, remote_path)
        debug(f"Persisted {label} to {remote_path}")
    except Exception as exc:
        debug(f"⚠️  Failed to persist {label} to {remote_path}: {exc}")


def _resolve_model_path(path: Path) -> Path:
    """Return a readable path for a VW model, falling back to local staging when needed."""

    def _normalize(candidate: Path) -> Path:
        candidate_str = str(candidate)
        if candidate_str.startswith("dbfs:/"):
            return Path("/dbfs" + candidate_str[5:])
        return candidate

    primary = _normalize(Path(path))
    candidates = [primary]

    # When Optuna passes a remote models dir but training staged locally, reuse the local copy.
    fallback_local = LOCAL_MODELS_DIR / primary.name
    if fallback_local not in candidates:
        candidates.append(fallback_local)

    fallback_remote = MODELS_DIR / primary.name
    if fallback_remote not in candidates:
        candidates.append(fallback_remote)

    for candidate in candidates:
        if candidate.exists():
            if candidate != primary:
                debug(f"Resolved model path mismatch: using {candidate} instead of missing {primary}")
            return candidate

    searched = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Model file not found. Checked paths: {searched}")


TRAIN_VW = _dbfs_to_local(DEFAULT_TRAIN_VW)
VALID_VW = _dbfs_to_local(DEFAULT_VALID_VW)
TEST_VW = _dbfs_to_local(DEFAULT_TEST_VW)
STREAMING_DIR = DATA_PROCESSED / "vw_streaming"


def ensure_vw_cli() -> None:
    """Ensure the vw binary is available on PATH; install if missing."""
    if shutil.which("vw"):
        debug("vw CLI already present on PATH")
        return

    debug("vw CLI not found. Installing vowpalwabbit==9.10.0 ...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--force-reinstall", "vowpalwabbit==9.10.0"],
        check=True,
    )

    candidates: List[Path] = []

    env_binary = os.environ.get("VW_BINARY")
    if env_binary:
        try:
            candidates.append(_dbfs_to_local(env_binary))
        except Exception:
            candidates.append(Path(env_binary))

    probe_script = "\n".join(
        [
            "import vowpalwabbit",
            "from pathlib import Path",
            "p = Path(vowpalwabbit.__file__).resolve().parent",
            "for candidate in p.rglob('vw'):",
            "    if candidate.is_file():",
            "        print(candidate)",
            "        break",
        ]
    )

    pkg_probe = subprocess.run(
        [sys.executable, "-c", probe_script],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    if pkg_probe:
        candidates.append(Path(pkg_probe))

    candidates.extend(
        [
            Path("/usr/local/bin/vw"),
            Path("/usr/bin/vw"),
            Path("/databricks/python/bin/vw"),
            Path("/databricks/python3/bin/vw"),
        ]
    )

    for candidate in candidates:
        try:
            candidate_path = Path(candidate)
            if candidate_path.is_file():
                candidate_path.chmod(0o755)
                subprocess.run(["ln", "-sf", str(candidate_path), "/usr/local/bin/vw"], check=True)
                debug(f"vw CLI linked from {candidate_path}")
                return
        except Exception as exc:
            debug(f"⚠️  Failed to use VW binary candidate {candidate}: {exc}")

    debug("⚠️  Attempting to install vw via apt-get ...")
    try:
        subprocess.run(["sudo", "apt-get", "update"], check=True)
        subprocess.run(["sudo", "apt-get", "install", "-y", "vowpal-wabbit"], check=True)
    except FileNotFoundError:
        debug("⚠️  apt-get not available on this runtime")
    except subprocess.CalledProcessError as exc:
        debug(f"⚠️  apt-get install failed: {exc}")

    vw_path = shutil.which("vw")
    if vw_path:
        Path(vw_path).chmod(0o755)
        subprocess.run(["ln", "-sf", vw_path, "/usr/local/bin/vw"], check=True)
        debug(f"vw CLI ready at {vw_path}")
        return

    raise RuntimeError(
        "vw binary not found after installation. Set VW_BINARY env var to an accessible path, "
        "or install vowpal-wabbit manually on the cluster."
    )



ensure_vw_cli()


def _normalize_dbfs_uri(path: str) -> str:
    if path.startswith("/dbfs/"):
        return "dbfs:/" + path[6:]
    return path


def _to_local_path(path: str) -> Path:
    if path.startswith("dbfs:/"):
        return Path("/dbfs" + path[5:])
    return Path(path)


def _concat_shards_to_file(src_dir: Path, dest_file: Path) -> None:
    """Concatenate part-* shards under src_dir into a single dest_file without loading into memory."""
    import glob
    shards = sorted(glob.glob(str(src_dir / 'part-*')))
    if not shards:
        raise FileNotFoundError(f"No shards found under {src_dir}")
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    with open(dest_file, 'wb') as out_f:
        for shard in shards:
            with open(shard, 'rb') as in_f:
                while True:
                    chunk = in_f.read(1024 * 1024)
                    if not chunk:
                        break
                    out_f.write(chunk)

def ensure_vw_files_from_streaming(streaming_root: Path,
                                   train_file: Path,
                                   valid_file: Path,
                                   test_file: Path) -> None:
    """Concatenate streaming shards into single VW files."""
    if not streaming_root.exists():
        raise FileNotFoundError(f"Streaming directory not found: {streaming_root}")

    def _prepare(split_name: str, dest: Path) -> None:
        split_dir = streaming_root / split_name
        if split_dir.exists():
            print(f"\n🔧 Concatenating {split_name} shards ({split_dir}) → {dest}")
            _concat_shards_to_file(split_dir, dest)
            print(f"   ✅ {split_name}.vw ready")
        else:
            print(f"ℹ️  {split_name.capitalize()} shards not found at {split_dir}; skipping.")

    train_file.parent.mkdir(parents=True, exist_ok=True)
    _prepare('train', train_file)
    _prepare('valid', valid_file)
    _prepare('test', test_file)


def build_vw_command(
    trial: optuna.Trial,
    train_file: Path,
    model_output: Path,
    cache_file: Optional[Path] = None,
) -> str:
    """
    Build VW command with hyperparameters suggested by Optuna.

    Args:
        trial: Optuna trial for hyperparameter suggestions
        train_file: Path to training VW file
        model_output: Path to save trained model
        cache_file: Optional cache file for faster training

    Returns:
        VW command string
    """
    # CB algorithm (fixed to DR)
    cb_type = "dr"

    # Core hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 0.5, log=True)
    power_t = trial.suggest_float('power_t', 0.0, 0.5)
    l2 = trial.suggest_float('l2', 1e-8, 1e-3, log=True)

    # Base command
    cmd_parts = [
        "vw",
        "--cb_explore_adf",
        f"--cb_type {cb_type}",
        f"--learning_rate {learning_rate}",
        f"--power_t {power_t}",
        f"--l2 {l2}",
        "-b 24",  # Increase bit precision to 24 (16M features) to prevent collisions
    ]

    # Exploration strategy (Epsilon Greedy only - Bagging is too slow for complex models)
    epsilon = trial.suggest_float('epsilon', 0.01, 0.2)
    cmd_parts.append(f"--epsilon {epsilon}")

    # Smart Interactions: Interact every User namespace with Action ('a')
    # Namespaces: u=User, s=Session, e=Economy, t=Temporal, l=Lag, w=EWMA, f=Feature
    # We want to learn: How does [User Context] affect [Action Reward]?
    # Strategy: Try different levels of interaction complexity
    interaction_level = trial.suggest_categorical(
        'interaction_level', ['basic', 'rich', 'full']
    )

    if interaction_level == 'basic':
        # Minimal: Just User x Action and Session x Action
        cmd_parts.append("-q ua -q sa")
    elif interaction_level == 'rich':
        # Rich: All main context groups x Action
        cmd_parts.append("-q ua -q sa -q ea -q wa")
    elif interaction_level == 'full':
        # Full: Every single context namespace x Action
        # u, s, e, t, l, w, f
        cmd_parts.append("-q ua -q sa -q ea -q ta -q la -q wa -q fa")

    # Training settings
    # More passes for complex surfaces to converge
    passes = trial.suggest_int('passes', 5, 10)
    cmd_parts.append(f"--passes {passes}")
    cmd_parts.append("--holdout_off")

    # Input/output
    cmd_parts.append(f"-d {train_file}")
    cmd_parts.append(f"-f {model_output}")

    if cache_file:
        cmd_parts.append(f"--cache_file {cache_file}")

    # Quiet mode (reduce output)
    cmd_parts.append("--quiet")

    return " ".join(cmd_parts)


def train_vw_model(vw_command: str, log_file: Path) -> Tuple[bool, str]:
    """
    Execute VW training command.

    Args:
        vw_command: VW command string
        log_file: Where to write training logs

    Returns:
        (success, log_output)
    """
    try:
        # Run VW
        result = subprocess.run(
            vw_command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=TRAIN_TIMEOUT_SECS
        )

        # Write log
        with open(log_file, 'w') as f:
            f.write(f"Command:\n{vw_command}\n\n")
            f.write(f"STDOUT:\n{result.stdout}\n\n")
            f.write(f"STDERR:\n{result.stderr}\n\n")
            f.write(f"Return code: {result.returncode}\n")

        # Check for specific VW errors that might occur even with exit code 0 (or masked by shell)
        # "your features have too much magnitude" -> VW aborts training
        if "your features have too much magnitude" in result.stderr:
            return False, f"VW Error: Features have too much magnitude. Check for unnormalized data or interactions.\nSTDERR: {result.stderr}"

        success = result.returncode == 0
        return success, result.stderr

    except subprocess.TimeoutExpired:
        return False, f"Training timeout ({TRAIN_TIMEOUT_SECS} seconds)"
    except Exception as e:
        return False, str(e)


def _load_logged_validation_data(validation_vw_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse validation VW file to extract logged rewards, propensities, and action indices."""
    global VALID_LOGGED_DATA
    if VALID_LOGGED_DATA is not None:
        return VALID_LOGGED_DATA

    valid_path = Path(validation_vw_path)
    if str(valid_path).startswith("dbfs:/"):
        valid_path = _dbfs_to_local(str(valid_path))

    rewards: List[float] = []
    propensities: List[float] = []
    chosen_indices: List[int] = []
    example_lines: List[str] = []

    def _flush_example(lines: List[str]) -> None:
        if not lines:
            return
        for action_line in lines[1:]:
            try:
                label_part, _ = action_line.split('|', 1)
            except ValueError:
                continue
            label_part = label_part.strip()
            if label_part.count(':') < 2:
                continue
            idx_str, cost_str, prob_str = label_part.split(':', 2)
            try:
                idx = int(idx_str)
                cost = float(cost_str)
                prob = float(prob_str)
            except ValueError:
                continue
            chosen_indices.append(idx)
            rewards.append(-cost)
            propensities.append(max(prob, 1e-6))
            return
        raise RuntimeError("Validation example missing logged propensity/cost")

    with open(valid_path, 'r', encoding='utf-8') as fh:
        for raw_line in fh:
            line = raw_line.rstrip('\n')


            if line.strip() == '':
                _flush_example(example_lines)
                example_lines = []
            else:
                example_lines.append(line)
        _flush_example(example_lines)

    if not rewards:
        raise RuntimeError("No validation examples parsed for IPS evaluation")

    VALID_LOGGED_DATA = (np.array(rewards, dtype=float),
                         np.array(propensities, dtype=float),
                         np.array(chosen_indices, dtype=int))
    debug(f"Loaded {len(rewards)} validation examples for IPS evaluation")
    return VALID_LOGGED_DATA



def _compute_policy_probabilities(model_path: Path, validation_vw_path: Path) -> np.ndarray:
    """Run VW in test mode to obtain policy action probabilities for each validation example."""
    model_path = _resolve_model_path(model_path)
    valid_path = Path(validation_vw_path)
    if str(valid_path).startswith("dbfs:/"):
        valid_path = _dbfs_to_local(str(valid_path))

    with tempfile.NamedTemporaryFile(mode="w+", delete=False) as tmp_pred:
        pred_path = Path(tmp_pred.name)

    cmd = [
        "vw",
        "-i", str(model_path),
        "-t",
        "-d", str(valid_path),
        "--json",
        "-p", str(pred_path),
        "--quiet",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)

    if result.returncode != 0:
        pred_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"VW probability prediction failed (exit {result.returncode}): {result.stderr.strip()}"
        )

    probs: List[np.ndarray] = []
    try:
        with open(pred_path, "r", encoding="utf-8") as fh:
            for raw_line in fh:
                record = raw_line.strip()
                if not record:
                    continue
                try:
                    data = json.loads(record)
                except json.JSONDecodeError:
                    continue

                action_scores = None
                if isinstance(data, dict):
                    if "action_scores" in data:
                        action_scores = data.get("action_scores")
                    elif "example" in data and isinstance(data["example"], dict):
                        action_scores = data["example"].get("action_scores")

                if not action_scores:
                    continue

                vec = np.zeros(len(ARM_ORDER), dtype=float)
                for score in action_scores:
                    try:
                        action_idx = int(score.get("action"))
                        prob_val = float(score.get("probability", 0.0))
                    except (TypeError, ValueError):
                        continue
                    if 0 <= action_idx < len(ARM_ORDER):
                        vec[action_idx] = max(prob_val, 0.0)

                total = vec.sum()
                if total <= 0:
                    vec[:] = 1.0 / len(ARM_ORDER)
                else:
                    vec /= total
                probs.append(vec)
    finally:
        # Persist JSON predictions for debugging before cleanup
        try:
            debug_pred_dir = ARTIFACTS_DIR / "predictions"
            debug_pred_dir.mkdir(parents=True, exist_ok=True)
            debug_pred_path = debug_pred_dir / f"{Path(model_path).stem}_valid_predictions.jsonl"
            shutil.copy2(pred_path, debug_pred_path)
        except Exception as exc:
            debug(f"⚠️  Failed to persist prediction JSON: {exc}")
        pred_path.unlink(missing_ok=True)

    if not probs:
        raise RuntimeError("VW probabilities output was empty")

    return np.vstack(probs)



def _estimate_ips_reward(model_path: Path, validation_vw_path: Path) -> float:
    """Compute an IPS estimate of next-day reward for the learned policy."""
    rewards, propensities, chosen_indices = _load_logged_validation_data(validation_vw_path)
    policy_probs = _compute_policy_probabilities(model_path, validation_vw_path)

    if policy_probs.shape[0] != len(rewards):
        raise RuntimeError(
            f"Prediction count {policy_probs.shape[0]} does not match validation examples {len(rewards)}"
        )

    unique_probs = np.unique(np.round(policy_probs, 4), axis=0)
    debug(f"Policy probability vectors: unique={len(unique_probs)} first={policy_probs[0][:5] if len(policy_probs) else []}")

    unique_props = np.unique(np.round(propensities, 4))
    debug(f"Logged propensities unique count={len(unique_props)} sample={unique_props[:5] if len(unique_props)>0 else []}")

    ips_terms = []
    for reward, logged_prop, chosen_idx, prob_vec in zip(rewards, propensities, chosen_indices, policy_probs):
        logged_p = max(logged_prop, 1e-6)
        pi = prob_vec[chosen_idx]
        ips_terms.append(reward * (pi / logged_p))

    if not ips_terms:
        raise RuntimeError("No IPS terms could be computed; check logged propensities")

    estimate = float(np.mean(ips_terms))
    variance = float(np.var(ips_terms))
    debug(
        f"IPS reward estimate={estimate:.2f} coins/day over {len(ips_terms)} examples (variance={variance:.2f})"
    )
    return estimate




def compute_progressive_validation_loss(
    model_path: Path,
    valid_file: Path,
) -> float:
    """Run VW in test mode to obtain the progressive validation loss."""
    model_path = _resolve_model_path(model_path)
    valid_path = Path(valid_file)
    if str(valid_path).startswith("dbfs:/"):
        valid_path = _dbfs_to_local(str(valid_path))

    cmd = [
        "vw",
        "-i", str(model_path),
        "-t",
        "-d", str(valid_path),
        "--progress", "1",
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
    except Exception as exc:
        debug(f"Failed to execute VW validation command: {exc}")
        raise

    output = "\n".join(filter(None, [result.stdout, result.stderr]))
    match_loss = re.search(r"average loss =\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)", output)
    if not match_loss:
        match_loss = re.search(r"avg loss =\s*([+-]?[0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)", output)
    if not match_loss:
        debug(f"VW output without parseable loss (exit={result.returncode}): {output[:400]}")
        raise RuntimeError("Could not parse VW validation loss from output")

    loss = float(match_loss.group(1))
    debug(f"Parsed VW validation loss={loss:.6f} for {valid_path}")
    return loss


def estimate_validation_dr_simple(
    model_path: Path,
    validation_vw_path: Path,
    precomputed_loss: Optional[float] = None,
) -> float:
    """Estimate expected next-day reward using IPS on the validation split."""
    try:
        if precomputed_loss is None:
            precomputed_loss = compute_progressive_validation_loss(model_path, validation_vw_path)

        ips_reward = _estimate_ips_reward(model_path, validation_vw_path)
        debug(
            f"Validation reward estimate={ips_reward:.2f} coins/day (baseline={OPE_UNIFORM_DR_MEAN:.2f}, "
            f"loss={precomputed_loss:.6f})"
        )
        return ips_reward

    except Exception as exc:
        debug(f"DR estimation error: {exc}")
        return -1e6

def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function.

    Returns:
        val_loss  # Progressive validation loss minimized by Optuna
    """
    trial_id = trial.number

    print(f"\n{'='*80}")
    print(f"Trial {trial_id}: Testing hyperparameters...")
    print(f"{'='*80}")

    # Create temp model file (local) and remote destination on DBFS
    temp_model_local = LOCAL_MODELS_DIR / f"vw_trial_{trial_id}.vw"
    cache_file_local = LOCAL_MODELS_DIR / f"train_cache_{trial_id}.cache"
    temp_model_remote = MODELS_DIR / f"vw_trial_{trial_id}.vw"
    log_file = LOGS_DIR / f"trial_{trial_id}.log"

    # Ensure scratch directories still exist (Databricks can wipe /local_disk0 mid-run)
    temp_model_local.parent.mkdir(parents=True, exist_ok=True)
    cache_file_local.parent.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    debug(f"Trial {trial_id} model path (local): {temp_model_local}")
    debug(f"Trial {trial_id} cache path (local): {cache_file_local}")

    try:
        # Build VW command
        vw_cmd = build_vw_command(
            trial, TRAIN_VW, temp_model_local, cache_file_local
        )

        print(f"  VW command:")
        print(f"    {vw_cmd[:200]}...")
        debug(f"Trial {trial_id} VW cmd: {vw_cmd}")

        # Train
        start_time = time.time()
        success, log_output = train_vw_model(vw_cmd, log_file)
        train_time = time.time() - start_time

        if not success:
            print(f"  ❌ Training failed: {log_output[:200]}")
            raise optuna.TrialPruned(f"Training failed")

        print(f"  ✅ Training completed in {train_time:.1f}s")
        if not _wait_for_path(temp_model_local, timeout=120):
            _debug_missing_model(temp_model_local, log_file)
            raise optuna.TrialPruned(f"Model file not found after training: {temp_model_local}")

        # Copy to DBFS before evaluation to survive local disk eviction
        _persist_file(temp_model_local, temp_model_remote, f"trial {trial_id} model")
        if not _wait_for_path(temp_model_remote, timeout=60):
            raise optuna.TrialPruned(f"Model file not visible on DBFS: {temp_model_remote}")

        # Evaluate on validation set (use remote model path)
        val_loss = compute_progressive_validation_loss(temp_model_remote, VALID_VW)

        print(f"  📊 Validation metrics:")
        print(f"    Loss: {val_loss:.4f}")

        # Cleanup temp files
        if temp_model_local.exists():
            temp_model_local.unlink()
        if cache_file_local.exists():
            cache_file_local.unlink()
        temp_model_remote.unlink(missing_ok=True)

        # Return objective (Optuna minimizes loss)
        return val_loss

    except Exception as e:
        print(f"  ❌ Trial failed: {e}")
        # Cleanup
        if temp_model_local.exists():
            temp_model_local.unlink()
        if cache_file_local.exists():
            cache_file_local.unlink()
        
        # Also clean up any stale .writing cache files left by aborted VW
        cache_writing = Path(str(cache_file_local) + ".writing")
        if cache_writing.exists():
            try:
                cache_writing.unlink()
                debug(f"Cleaned up stale cache file: {cache_writing}")
            except Exception as cleanup_err:
                debug(f"Failed to clean up {cache_writing}: {cleanup_err}")

        temp_model_remote.unlink(missing_ok=True)
        raise optuna.TrialPruned(str(e))


def run_optuna_optimization(
    n_trials: int = 100,
    n_jobs: int = 1,
    study_name: str = "vw_dr_bandit",
) -> optuna.Study:
    """
    Run Optuna hyperparameter optimization.

    Args:
        n_trials: Number of trials
        n_jobs: Parallel jobs (1 = sequential, -1 = all cores)
        study_name: Study name

    Returns:
        Completed Optuna study
    """
    print("\n" + "=" * 80)
    print("OPTUNA HYPERPARAMETER OPTIMIZATION")
    print("=" * 80)

    if REMOTE_OPTUNA_DB.exists() and not LOCAL_OPTUNA_DB.exists():
        try:
            shutil.copy2(REMOTE_OPTUNA_DB, LOCAL_OPTUNA_DB)
            debug(f"Restored Optuna study DB from {REMOTE_OPTUNA_DB}")
        except Exception as exc:
            debug(f"⚠️  Failed to restore Optuna DB from {REMOTE_OPTUNA_DB}: {exc}")

    # Create study
    storage = f"sqlite:///{LOCAL_OPTUNA_DB}"

    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )

    # Legacy safeguard: if an existing multi-objective study is loaded, spin up a new single-objective study.
    active_study_name = study_name
    if len(study.directions) != 1:
        print("\n⚠️  Existing study is multi-objective. Creating a new single-objective study for loss-only tuning.")
        suffix = "loss_only"
        attempt = 1
        while True:
            candidate_name = f"{study_name}_{suffix}" if attempt == 1 else f"{study_name}_{suffix}_{attempt}"
            try:
                study = optuna.create_study(
                    study_name=candidate_name,
                    storage=storage,
                    direction="minimize",
                    load_if_exists=False,
                    sampler=optuna.samplers.TPESampler(seed=42),
                    pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
                )
                active_study_name = candidate_name
                break
            except optuna.exceptions.DuplicatedStudyError:
                attempt += 1

        print(f"   ➜ Using new study: {active_study_name}")
    else:
        active_study_name = study.study_name

    print(f"\n🎯 Optimization settings:")
    print(f"  Study: {active_study_name}")
    print(f"  Trials: {n_trials}")
    print(f"  Jobs: {n_jobs}")
    print(f"  Storage (local): {storage}")
    print(f"  Remote copy: {REMOTE_OPTUNA_DB}")
    # Run optimization
    print(f"\n🚀 Starting optimization...")
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)

    print(f"\n✅ Optimization complete!")
    print(f"  Completed trials: {len(study.trials)}")

    _persist_file(LOCAL_OPTUNA_DB, REMOTE_OPTUNA_DB, "Optuna study DB")

    return study


def select_best_trial(study: optuna.Study) -> optuna.Trial:
    """Select the trial with the lowest validation loss."""
    debug("Evaluating Optuna trials to select best model")
    print("\n" + "=" * 80)
    print("SELECTING BEST TRIAL")
    print("=" * 80)

    try:
        best_trial = study.best_trial
    except ValueError:
        completed_trials = [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
        ]
        if not completed_trials:
            state_counts = Counter(t.state for t in study.trials)
            debug("No completed Optuna trials with objective values")
            print("\n  ⚠️  No completed Optuna trials with objective values.")
            print("  Trial states:")
            for state, count in sorted(state_counts.items(), key=lambda kv: kv[0].name):
                print(f"    - {state.name:<12}: {count}")
            raise RuntimeError(
                "No successful Optuna trials. Check VW logs under logs/training, loosen pruning criteria, "
                "or increase n_trials to ensure at least one trial completes."
            )
        best_trial = min(completed_trials, key=lambda t: t.value)

    debug(f"Selected trial {best_trial.number} with loss={best_trial.value:.4f}")

    leaderboard = sorted(
        [
            t for t in study.trials
            if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
        ],
        key=lambda t: t.value
    )[:10]
    if leaderboard:
        print(f"\n  Top {len(leaderboard)} trials by validation loss:")
        for rank, trial in enumerate(leaderboard, 1):
            print(f"    {rank}. Trial {trial.number}: loss={trial.value:.4f}")

    print(f"\n  ✅ Selected trial {best_trial.number}:")
    print(f"    Validation loss: {best_trial.value:.4f}")
    print(f"\n  Hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"    {key:20s}: {value}")

    return best_trial


def train_final_model(best_params: Dict[str, Any]) -> Path:
    """Train final VW model using the best hyperparameters and persist to DBFS."""
    print("\n" + "=" * 80)
    print("TRAINING FINAL MODEL")
    print("=" * 80)

    class MockTrial:
        def __init__(self, params: Dict[str, Any]) -> None:
            self.params = params

        def suggest_categorical(self, name: str, choices: List[Any]) -> Any:
            return self.params.get(name, choices[0])

        def suggest_float(self, name: str, low: float, high: float, log: bool = False) -> float:
            return float(self.params.get(name, (low + high) / 2))

        def suggest_int(self, name: str, low: int, high: int) -> int:
            return int(self.params.get(name, (low + high) // 2))

    mock_trial = MockTrial(best_params)

    # Generate timestamp for model versioning
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y_%m_%d')
    
    final_model_path = MODELS_DIR / f"vw_bandit_dr_{timestamp}.vw"
    local_model_path = LOCAL_MODELS_DIR / f"vw_bandit_dr_{timestamp}_local.vw"
    cache_path = LOCAL_MODELS_DIR / "train_final.cache"
    final_model_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        cache_path.unlink()

    vw_cmd = build_vw_command(mock_trial, TRAIN_VW, local_model_path, cache_path)

    print(f"\n  VW command:")
    print(f"    {vw_cmd}")

    log_file = LOGS_DIR / "final_model_training.log"
    success, log_output = train_vw_model(vw_cmd, log_file)

    if not success:
        raise RuntimeError(f"Final model training failed: {log_output}")

    if not _wait_for_path(local_model_path, timeout=180):
        raise RuntimeError(f"Final model file not found after training: {local_model_path}")

    _persist_file(local_model_path, final_model_path, "final vw model")

    if cache_path.exists():
        cache_path.unlink()
    local_model_path.unlink(missing_ok=True)

    print(f"\n  ✅ Final model trained: {final_model_path}")
    
    # Cleanup old models (keep last 10)
    cleanup_old_models(MODELS_DIR, keep_last=10)

    return final_model_path


def cleanup_old_models(models_dir: Path, pattern: str = "vw_bandit_dr_*.vw", keep_last: int = 10):
    """
    Keep only the last N models, delete older ones.
    
    Args:
        models_dir: Directory containing models
        pattern: Glob pattern for model files
        keep_last: Number of models to keep
    """
    # Convert DBFS path to local if needed
    if str(models_dir).startswith('dbfs:/'):
        models_dir = Path(str(models_dir).replace('dbfs:/', '/dbfs/'))
    
    # Find all matching models
    models = list(models_dir.glob(pattern))
    
    if len(models) <= keep_last:
        print(f"   ℹ️  Found {len(models)} models, keeping all (threshold: {keep_last})")
        return  # Nothing to clean
    
    # Sort by modification time (oldest first)
    models.sort(key=lambda p: p.stat().st_mtime)
    
    # Delete oldest models
    to_delete = models[:len(models) - keep_last]
    for model in to_delete:
        print(f"   🗑️  Deleting old model: {model.name}")
        model.unlink()
    
    print(f"   ✅ Model cleanup: Kept last {keep_last} models, deleted {len(to_delete)} old models")



def save_best_config(best_params: Dict[str, Any]) -> Path:
    """Save best hyperparameters to YAML locally and on DBFS."""
    config = {
        "cb_type": "dr",
        **best_params,
        "model_path": str(MODELS_DIR / "vw_bandit_dr_best.vw"),
    }

    local_config = LOCAL_BASE_DIR / "config" / "vw_config_best.yaml"
    remote_config = CONFIG_DIR / "vw_config_best.yaml"

    local_config.parent.mkdir(parents=True, exist_ok=True)
    with open(local_config, "w", encoding="utf-8") as fh:
        yaml.dump(config, fh, default_flow_style=False, sort_keys=False)

    _persist_file(local_config, remote_config, "best config YAML")
    return remote_config



def create_optuna_visualizations(study: optuna.Study) -> None:
    """Generate Optuna visualizations and sync to DBFS."""
    print("\n" + "=" * 80)
    print("CREATING VISUALIZATIONS")
    print("=" * 80)

    try:
        import plotly
    except ImportError:
        print("  ⚠️  plotly not installed, skipping visualizations")
        return

    artifacts = {}

    # Optimization history
    try:
        fig = plot_optimization_history(study)
        artifacts["optuna_optimization_history.html"] = fig
    except Exception as exc:
        debug(f"⚠️  Failed to create optimization history: {exc}")

    # Parameter importances
    try:
        fig = plot_param_importances(study)
        artifacts["optuna_param_importances.html"] = fig
    except Exception as exc:
        debug(f"⚠️  Failed to create param importances: {exc}")

    if not artifacts:
        debug("No visualizations were generated")
        return

    for name, figure in artifacts.items():
        local_path = LOCAL_ARTIFACTS_DIR / name
        try:
            figure.write_html(str(local_path))
            _persist_file(local_path, ARTIFACTS_DIR / name, f"Optuna visualization {name}")
        except Exception as exc:
            debug(f"⚠️  Failed to persist visualization {name}: {exc}")



def main() -> None:
    """Run VW training with Optuna optimization."""

    global STREAMING_DIR, TRAIN_VW, VALID_VW, TEST_VW

    debug("Phase 3 main() starting")
    print("\n" + "=" * 80)
    print(" " * 15 + "VW CONTEXTUAL BANDIT TRAINING")
    print(" " * 20 + "WITH OPTUNA OPTIMIZATION")
    print(" " * 30 + "PHASE 3")
    print("=" * 80)

    parser = argparse.ArgumentParser(description="VW contextual bandit training with Optuna")
    parser.add_argument('--streaming-dir', type=str, default='', help='Directory containing train/valid/test shards (DBFS or local).')
    parser.add_argument('--train-file', type=str, default=str(TRAIN_VW), help='Output path for concatenated train VW file.')
    parser.add_argument('--valid-file', type=str, default=str(VALID_VW), help='Output path for concatenated validation VW file.')
    parser.add_argument('--test-file', type=str, default=str(TEST_VW), help='Output path for concatenated test VW file.')
    parser.add_argument('--n-trials', type=int, default=100, help='Number of Optuna trials to run.')
    parser.add_argument('--n-jobs', type=int, default=1, help='Parallel jobs for Optuna (1 = sequential, -1 = all cores).')
    parser.add_argument('--study-name', type=str, default='vw_dr_bandit', help='Optuna study name.')
    
    # Training pipeline chaining
    parser.add_argument('--trigger-phase4', action='store_true', help='Trigger Phase 4 after Phase 3 completes')
    parser.add_argument('--phase4-job-id', type=int, default=738622390983426, help='Databricks Job ID for Phase 4')
    
    args = parser.parse_args()

    debug(f"Parsed args: streaming_dir={args.streaming_dir}, train_file={args.train_file}, "
          f"valid_file={args.valid_file}, test_file={args.test_file}, n_trials={args.n_trials}, "
          f"n_jobs={args.n_jobs}, study_name={args.study_name}")

    streaming_dir_arg = args.streaming_dir.strip()
    STREAMING_DIR = _to_local_path(streaming_dir_arg) if streaming_dir_arg else STREAMING_DIR
    TRAIN_VW = _to_local_path(args.train_file)
    VALID_VW = _to_local_path(args.valid_file)
    TEST_VW = _to_local_path(args.test_file)

    debug(f"Resolved paths -> STREAMING_DIR={STREAMING_DIR}, TRAIN_VW={TRAIN_VW}, "
          f"VALID_VW={VALID_VW}, TEST_VW={TEST_VW}")

    try:
        print(f"\n📁 Streaming shards root: {STREAMING_DIR}")
        ensure_vw_files_from_streaming(STREAMING_DIR, TRAIN_VW, VALID_VW, TEST_VW)
        debug("Completed ensure_vw_files_from_streaming")

        if not TRAIN_VW.exists():
            raise FileNotFoundError(f"Run Phase 2 first! Missing train data at {TRAIN_VW}")
        if not VALID_VW.exists():
            raise FileNotFoundError(f"Run Phase 2 first! Missing validation data at {VALID_VW}")
        debug("Verified train/valid VW files exist")

        print(f"\n✅ Input files verified:")
        print(f"  Train: {TRAIN_VW}")
        print(f"  Valid: {VALID_VW}")

        debug("Starting Optuna optimization")
        study = run_optuna_optimization(
            n_trials=args.n_trials,
            n_jobs=args.n_jobs,
            study_name=args.study_name
        )
        debug(f"Optuna study complete: {len(study.trials)} total trials")

        best_trial = select_best_trial(study)

        final_model_path = train_final_model(best_trial.params)
        debug(f"Final model stored at {final_model_path}")

        save_best_config(best_trial.params)
        debug("Saved best config")

        create_optuna_visualizations(study)
        debug("Generated Optuna visualizations (if plotly installed)")

        # MLflow logging (optional - requires mlflow package and env var MLFLOW_EXPERIMENT_PATH)
        if not MLFLOW_AVAILABLE:
            debug("⚠️  MLflow not available. Skipping MLflow logging. Install mlflow to enable.")
        else:
            try:
                exp_path = os.environ.get("MLFLOW_EXPERIMENT_PATH", "").strip()
                model_name = os.environ.get("MLFLOW_MODEL_NAME", "").strip()
                if exp_path:
                    print("\n" + "=" * 80)
                    print("MLFLOW LOGGING")
                    print("=" * 80)
                    mlflow.set_experiment(exp_path)
                    with mlflow.start_run(run_name=f"phase3-{args.study_name}"):
                        # Log best hyperparameters
                        for k, v in best_trial.params.items():
                            mlflow.log_param(k, v)
                        # Compute and log final validation loss for the trained model (best config)
                        try:
                            final_loss = compute_progressive_validation_loss(_dbfs_to_local(str(MODELS_DIR / "vw_bandit_dr_best.vw")), VALID_VW)
                            mlflow.log_metric("validation_loss", final_loss)
                        except Exception as exc:
                            debug(f"Could not compute/log final validation loss: {exc}")
                        # Log artifacts: raw VW model + config
                        model_info = None
                        try:
                            local_vw_path = str(_dbfs_to_local(str(MODELS_DIR / "vw_bandit_dr_best.vw")))
                            mlflow.log_artifact(local_vw_path, artifact_path="vw_model")
                        except Exception as exc:
                            debug(f"Could not log VW model artifact: {exc}")
                        try:
                            mlflow.log_artifact(str(_dbfs_to_local(str(CONFIG_DIR / 'vw_config_best.yaml'))), artifact_path="config")
                        except Exception as exc:
                            debug(f"Could not log config YAML artifact: {exc}")

                        # Log a pyfunc-wrapped VW model for serving
                        try:
                            model_info = mlflow.pyfunc.log_model(
                                artifact_path="model",
                                python_model=VWModelWrapper(),
                                artifacts={"vw_model": local_vw_path},
                            )
                            debug(f"Logged MLflow pyfunc model at {model_info.model_uri}")
                        except Exception as exc:
                            debug(f"Could not log MLflow pyfunc model: {exc}")

                        # Optional: register model in Model Registry using the pyfunc model URI
                        if model_name and model_info is not None:
                            try:
                                # Use legacy Workspace Model Registry (no Unity Catalog path required)
                                mlflow.set_registry_uri("databricks")
                                registered = mlflow.register_model(
                                    model_uri=model_info.model_uri,
                                    name=model_name,
                                )
                                print(f"  ✅ Registered model '{model_name}' as version {registered.version}")
                            except Exception as exc:
                                print(f"  ⚠️  Model registration failed: {exc}")
                    print("  ✅ MLflow logging complete")
                else:
                    debug("MLFLOW_EXPERIMENT_PATH not set. Skipping MLflow logging.")
            except Exception as exc:
                debug(f"MLflow logging skipped/failed: {exc}")

        print("\n" + "=" * 80)
        print("TRAINING COMPLETE!")
        print("=" * 80)
        print(f"\n✅ Best model: {final_model_path}")
        print(f"✅ Best config: {CONFIG_DIR / 'vw_config_best.yaml'}")
        print(f"✅ Study database: {REMOTE_OPTUNA_DB}")
        print(f"\n🎯 Next step: Run 04_validate_vw.py to validate model performance")
        print("=" * 80 + "\n")
        
        # Trigger Phase 4 if requested
        if args.trigger_phase4:
            print("\n" + "=" * 80)
            print("TRAINING PIPELINE: TRIGGERING PHASE 4")
            print("=" * 80)
            
            try:
                import requests
                
                # Get Databricks token
                databricks_token = os.environ.get("DATABRICKS_TOKEN")
                if not databricks_token:
                    print("   ⚠️  No DATABRICKS_TOKEN found; skipping Phase 4 trigger")
                else:
                    # Get workspace URL
                    workspace_url = os.environ.get("DATABRICKS_WORKSPACE_URL", "https://adb-249008710733422.2.azuredatabricks.net")
                    if not workspace_url.startswith("http"):
                        workspace_url = f"https://{workspace_url}"
                    
                    # Trigger Phase 4 job
                    api_url = f"{workspace_url}/api/2.1/jobs/run-now"
                    payload = {
                        "job_id": args.phase4_job_id,
                        "python_params": [
                            "--model-path", str(final_model_path),  # Use timestamped model
                            "--streaming-dir", args.streaming_dir,
                            "--propensity-spark-model", "dbfs:/mnt/models/propensity_spark",
                            "--bootstrap-samples", "1000"
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
                            
                            print(f"   ✅ Triggered Phase 4 (Job ID: {args.phase4_job_id}, Run ID: {run_id})")
                            logging.info(f"Triggered Phase 4 (Job ID: {args.phase4_job_id}, Run ID: {run_id})")
                            print(f"   🔗 View run: {workspace_url}/#job/{args.phase4_job_id}/run/{run_id}")
                            print(f"   📊 Training Pipeline: Phase 3 → Phase 4 (Validate Model)")
                            print(f"   🎉 Training pipeline complete: Phase 0 → 1 → 2 → 3 → 4")
                            break  # Success, exit retry loop
                            
                        except Exception as retry_error:
                            if attempt < max_retries:
                                print(f"   🔄 Retry {attempt + 1}/{max_retries} after error: {retry_error}")
                                logging.warning(f"Phase 4 trigger attempt {attempt + 1} failed: {retry_error}")
                                print(f"   ⏳ Waiting 5 seconds before retry...")
                                import time
                                time.sleep(5)
                            else:
                                print(f"   ❌ Failed to trigger Phase 4 after {max_retries + 1} attempts: {retry_error}")
                                logging.error(f"Phase 4 trigger failed after {max_retries + 1} attempts: {retry_error}")
                    
            except Exception as e:
                print(f"   ⚠️  Failed to trigger Phase 4: {e}")
                import traceback
                traceback.print_exc()
        
        debug("Phase 3 main() completed successfully")

    except Exception as exc:
        debug(f"Phase 3 main() failed: {exc}")
        raise
    finally:
        _persist_file(DEBUG_LOG, REMOTE_DEBUG_LOG, "Phase 3 debug log")
        
        # Persist the main pipeline log
        try:
            if os.path.exists(LOCAL_PIPELINE_LOG):
                os.makedirs(os.path.dirname(REMOTE_PIPELINE_LOG), exist_ok=True)
                shutil.copy2(LOCAL_PIPELINE_LOG, REMOTE_PIPELINE_LOG)
                print(f"[Phase3] Persisted pipeline log to {REMOTE_PIPELINE_LOG}")
        except Exception as e:
            print(f"[Phase3] ⚠️ Failed to persist pipeline log: {e}")


if __name__ == "__main__":
    main()
