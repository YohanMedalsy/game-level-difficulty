"""
FastAPI Inference Service for VW Contextual Bandit

Production-ready REST API for real-time difficulty predictions.

Features:
- <10ms p99 latency
- Health checks and readiness probes
- Prometheus metrics
- Graceful degradation (fallback to uniform policy)
- Request logging for online learning
- Feature extraction from user context

Endpoints:
- POST /predict - Get difficulty recommendation
- GET /health - Health check
- GET /metrics - Prometheus metrics
"""

import sys
import os
import time
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import pickle
import logging

import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

# Add parent to path
BANDIT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(BANDIT_ROOT / "src"))

from constants import (
    ACTION_TO_DELTA,
    ARM_ORDER,
    get_valid_arms,
    compute_next_multiplier,
    assign_feature_to_namespace,
)

# Paths
MODELS_DIR = BANDIT_ROOT / "models"
ARTIFACTS_DIR = BANDIT_ROOT / "artifacts"
LOGS_DIR = BANDIT_ROOT / "logs" / "inference"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / f"api_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Prometheus metrics
REQUESTS_TOTAL = Counter('bandit_requests_total', 'Total prediction requests')
REQUESTS_FAILED = Counter('bandit_requests_failed', 'Failed prediction requests')
PREDICTION_LATENCY = Histogram('bandit_prediction_latency_seconds', 'Prediction latency')
ACTIVE_REQUESTS = Gauge('bandit_active_requests', 'Currently active requests')
FALLBACK_PREDICTIONS = Counter('bandit_fallback_predictions', 'Predictions using fallback policy')

# FastAPI app
app = FastAPI(
    title="VW Contextual Bandit API",
    description="Real-time difficulty recommendations using Vowpal Wabbit",
    version="1.0.0"
)

# Global state (loaded on startup)
vw_model_path: Optional[Path] = None
selected_features: Optional[List[str]] = None
propensity_model = None
feature_scaler = None
namespace_features: Optional[Dict[str, List[str]]] = None

# Databricks Model Serving configuration
# When running inside Databricks, token is optional (uses internal auth)
# When running outside, set DATABRICKS_TOKEN env var
DATABRICKS_ENDPOINT_URL = os.getenv(
    "DATABRICKS_ENDPOINT_URL",
    "https://adb-249008710733422.2.azuredatabricks.net/serving-endpoints/vw-bandit-dr-endpoint/invocations"
)
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "")

# Check if running inside Databricks (has DATABRICKS_RUNTIME_VERSION)
IS_DATABRICKS = "DATABRICKS_RUNTIME_VERSION" in os.environ


# ============================================================================
# Request/Response Models
# ============================================================================

class UserContext(BaseModel):
    """User context for prediction."""
    user_id: str = Field(..., description="User ID")
    current_effectivelevelmultiplier: float = Field(..., description="Current difficulty multiplier")

    # Feature values (50 selected features)
    features: Dict[str, float] = Field(..., description="Feature values (50 selected features)")

    class Config:
        json_schema_extra = {
            "example": {
                "user_id": "user_12345",
                "current_effectivelevelmultiplier": 0.88,
                "features": {
                    "user_lifetime_days": 45,
                    "session_count_7d": 12,
                    "coins_balance": 1500,
                    # ... (47 more features)
                }
            }
        }


class PredictionResponse(BaseModel):
    """Prediction response."""
    user_id: str
    chosen_arm: str
    arm_probabilities: Dict[str, float]
    next_multiplier: float
    current_multiplier: float
    feasible_arms: List[str]
    prediction_time_ms: float
    model_version: str
    fallback_used: bool = False


# ============================================================================
# Startup/Shutdown
# ============================================================================

@app.on_event("startup")
async def load_models():
    """Load artifacts and verify Databricks endpoint configuration on startup."""
    global vw_model_path, selected_features, propensity_model, feature_scaler, namespace_features

    logger.info("Loading artifacts and verifying Databricks endpoint...")

    # Verify Databricks endpoint configuration
    if IS_DATABRICKS:
        logger.info(f"Running inside Databricks. Endpoint: {DATABRICKS_ENDPOINT_URL}")
        # Inside Databricks, we can use internal auth or no token for same-workspace calls
        # Token is optional but recommended for production
        if DATABRICKS_TOKEN:
            logger.info("Using provided DATABRICKS_TOKEN for endpoint auth")
        else:
            logger.info("No DATABRICKS_TOKEN set - will attempt internal Databricks auth")
    else:
        if not DATABRICKS_TOKEN:
            logger.warning("DATABRICKS_TOKEN not set and not running in Databricks. Fallback mode only.")
        else:
            logger.info(f"Databricks endpoint configured: {DATABRICKS_ENDPOINT_URL}")
    
    # Test endpoint connectivity (optional, can be removed if too slow)
    if DATABRICKS_ENDPOINT_URL:
        try:
            headers = {"Content-Type": "application/json"}
            if DATABRICKS_TOKEN:
                headers["Authorization"] = f"Bearer {DATABRICKS_TOKEN}"
            elif IS_DATABRICKS:
                # Try to get token from Databricks context (if available)
                try:
                    from pyspark.dbutils import DBUtils
                    dbutils = DBUtils()
                    token = dbutils.secrets.get(scope="databricks", key="token") if hasattr(dbutils, 'secrets') else None
                    if token:
                        headers["Authorization"] = f"Bearer {token}"
                        logger.info("Using Databricks secret for auth")
                except Exception:
                    pass  # No secret available, will try without auth
            
            test_response = requests.post(
                DATABRICKS_ENDPOINT_URL,
                headers=headers,
                json={"inputs": ["shared |u test:1\n0 |a arm:0"]},
                timeout=5,
            )
            if test_response.status_code == 200:
                logger.info("✅ Databricks endpoint is reachable")
            else:
                logger.warning(f"⚠️  Databricks endpoint returned status {test_response.status_code}")
        except Exception as e:
            logger.warning(f"⚠️  Could not verify Databricks endpoint: {e}")

    # Keep local model path for fallback (optional, if you want local fallback)
    vw_model_path = MODELS_DIR / "vw_bandit_dr_best.vw"
    if vw_model_path.exists():
        logger.info(f"Local VW model found (fallback): {vw_model_path}")

    # Load selected features
    # Priority: 1) SELECTED_FEATURES_PATH env var, 2) DBFS default, 3) Local repo
    features_path_str = os.getenv("SELECTED_FEATURES_PATH")
    
    if features_path_str:
        # Environment variable specified (could be DBFS path)
        if features_path_str.startswith("dbfs:/"):
            # Convert DBFS path to local mount path
            features_path = Path("/dbfs" + features_path_str[5:])
        else:
            features_path = Path(features_path_str)
        logger.info(f"Using features path from SELECTED_FEATURES_PATH env var: {features_path}")
    else:
        # Try DBFS default location first
        dbfs_default = Path("/dbfs/mnt/artifacts/selected_features_50.json")
        if dbfs_default.exists():
            features_path = dbfs_default
            logger.info(f"Using DBFS default features path: {features_path}")
        else:
            # Fall back to local repo
            features_path = ARTIFACTS_DIR / "selected_features_50.json"
            logger.info(f"Using local repo features path: {features_path}")
    
    if features_path.exists():
        with open(features_path, 'r') as f:
            feature_data = json.load(f)
        selected_features = feature_data['selected_features']
        logger.info(f"✅ Loaded {len(selected_features)} selected features from {features_path}")

        # Organize into namespaces
        namespace_features = {}
        from collections import defaultdict
        ns_dict = defaultdict(list)
        for feature in selected_features:
            ns = assign_feature_to_namespace(feature)
            ns_dict[ns].append(feature)
        namespace_features = dict(ns_dict)
        logger.info(f"Organized into {len(namespace_features)} namespaces")
    else:
        logger.warning(f"Selected features not found: {features_path}")

    # Load propensity model
    propensity_path = MODELS_DIR / "propensity_model.pkl"
    if propensity_path.exists():
        with open(propensity_path, 'rb') as f:
            propensity_model = pickle.load(f)
        logger.info("Propensity model loaded")
    else:
        logger.warning(f"Propensity model not found: {propensity_path}")

    # Load feature scaler
    scaler_path = MODELS_DIR / "feature_scaler.pkl"
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            feature_scaler = pickle.load(f)
        logger.info("Feature scaler loaded")
    else:
        logger.warning(f"Feature scaler not found: {scaler_path}")

    logger.info("✅ Startup complete")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    logger.info("Shutting down API...")


# ============================================================================
# Helper Functions
# ============================================================================

def extract_feature_vector(context: UserContext) -> np.ndarray:
    """
    Extract feature vector from user context.

    Returns:
        Feature vector (50 features in correct order)
    """
    if selected_features is None:
        raise ValueError("Selected features not loaded")

    feature_values = []
    for feature in selected_features:
        value = context.features.get(feature, 0.0)
        feature_values.append(value)

    return np.array(feature_values).reshape(1, -1)


def format_vw_example(
    context: UserContext,
    feature_vector: np.ndarray,
) -> str:
    """
    Format user context as VW example for prediction.

    Returns:
        VW CB_ADF formatted string
    """
    if namespace_features is None or selected_features is None:
        raise ValueError("Namespace features not loaded")

    # Build shared context
    shared_parts = []
    for namespace, features in sorted(namespace_features.items()):
        ns_parts = []
        for feature in features:
            idx = selected_features.index(feature)
            value = feature_vector[0, idx]
            if value != 0:  # Skip zero values for efficiency
                clean_name = feature.replace(' ', '_').replace(':', '_').replace('|', '_')
                ns_parts.append(f"{clean_name}:{value:.6f}")

        if ns_parts:
            shared_parts.append(f"|{namespace} " + " ".join(ns_parts))

    shared_line = "shared " + " ".join(shared_parts)

    # Build action lines (without costs/probabilities for prediction)
    current_mult = context.current_effectivelevelmultiplier
    valid_arms = get_valid_arms(current_mult)

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


def predict_with_vw(vw_example: str, feasible_arms: Optional[List[str]] = None) -> Tuple[str, Dict[str, float]]:
    """
    Run VW prediction via Databricks Model Serving endpoint.

    Returns:
        (chosen_arm, arm_probabilities)
    """
    if not DATABRICKS_ENDPOINT_URL:
        raise ValueError("DATABRICKS_ENDPOINT_URL not configured")

    # Call Databricks Model Serving endpoint
    # The pyfunc wrapper expects inputs as a list of VW examples (strings)
    payload = {"inputs": [vw_example]}
    
    headers = {"Content-Type": "application/json"}
    
    # Add auth header if token is available
    if DATABRICKS_TOKEN:
        headers["Authorization"] = f"Bearer {DATABRICKS_TOKEN}"
    elif IS_DATABRICKS:
        # Try to get token from Databricks context (if available)
        try:
            from pyspark.dbutils import DBUtils
            dbutils = DBUtils()
            token = dbutils.secrets.get(scope="databricks", key="token") if hasattr(dbutils, 'secrets') else None
            if token:
                headers["Authorization"] = f"Bearer {token}"
        except Exception:
            pass  # No secret available, will try without auth (may work for same-workspace calls)

    try:
        response = requests.post(
            DATABRICKS_ENDPOINT_URL,
            headers=headers,
            json=payload,
            timeout=5,  # 5 second timeout
        )
        response.raise_for_status()
        
        # Parse response
        # Databricks Model Serving returns: {"predictions": [<prediction>]} or {"outputs": [...]}
        # For our pyfunc wrapper, prediction is a float (action index 0-4)
        result = response.json()
        
        # Handle different response formats
        if "predictions" in result:
            predictions = result["predictions"]
        elif "outputs" in result:
            predictions = result["outputs"]
        elif isinstance(result, list):
            predictions = result
        else:
            # Try to extract first value if it's a dict with numeric values
            predictions = [v for v in result.values() if isinstance(v, (int, float, list))]
            if predictions and isinstance(predictions[0], list):
                predictions = predictions[0]
        
        if not predictions or len(predictions) == 0:
            raise ValueError(f"Empty predictions from Databricks endpoint. Response: {result}")
        
        # The pyfunc wrapper returns a pandas Series, which gets serialized as a list
        # Each element is a float (action index 0-4)
        pred_value = float(predictions[0]) if isinstance(predictions[0], (int, float)) else float(predictions[0][0] if isinstance(predictions[0], list) else predictions[0])
        
        # If it's an integer (action index), convert to probabilities
        # If it's a float between 0-1, it might be a probability for one arm
        # For now, assume it's an action index (0-4)
        if 0 <= pred_value <= 4 and pred_value == int(pred_value):
            # It's an action index
            chosen_idx = int(pred_value)
            probs = {arm: 0.0 for arm in ARM_ORDER}
            probs[ARM_ORDER[chosen_idx]] = 1.0
        else:
            # It's a probability or something else - fallback to uniform
            logger.warning(f"Unexpected prediction value from endpoint: {pred_value}, using uniform")
            probs = {arm: 1.0 / len(ARM_ORDER) for arm in ARM_ORDER}
            chosen_idx = 0
        
        # Apply feasibility: zero-out infeasible arms and renormalize
        if feasible_arms is not None and len(feasible_arms) > 0:
            for arm in ARM_ORDER:
                if arm not in feasible_arms:
                    probs[arm] = 0.0
            total = sum(probs.values())
            if total > 0:
                probs = {k: (v / total) for k, v in probs.items()}
            else:
                # all zero? uniform over feasible
                u = 1.0 / len(feasible_arms)
                probs = {arm: (u if arm in feasible_arms else 0.0) for arm in ARM_ORDER}
                chosen_idx = ARM_ORDER.index(feasible_arms[0])  # pick first feasible

        # Sample action according to probabilities
        arms = np.array(ARM_ORDER)
        pvals = np.array([probs[a] for a in arms])
        if pvals.sum() <= 0:
            # uniform safeguard
            pvals = np.ones(len(arms)) / len(arms)
        chosen_idx = int(np.random.choice(np.arange(len(arms)), p=pvals))
        chosen_arm = ARM_ORDER[chosen_idx]

        return chosen_arm, probs

    except requests.exceptions.RequestException as e:
        logger.error(f"Databricks endpoint request failed: {e}")
        raise RuntimeError(f"Databricks endpoint error: {e}")
    except (KeyError, ValueError, IndexError) as e:
        logger.error(f"Failed to parse Databricks response: {e}, response: {response.text if 'response' in locals() else 'N/A'}")
        raise RuntimeError(f"Invalid response from Databricks endpoint: {e}")


def predict_fallback(context: UserContext) -> Tuple[str, Dict[str, float]]:
    """
    Fallback prediction: uniform random over feasible arms.

    Returns:
        (chosen_arm, arm_probabilities)
    """
    valid_arms = get_valid_arms(context.current_effectivelevelmultiplier)

    # Uniform probabilities
    uniform_prob = 1.0 / len(valid_arms)
    arm_probs = {arm: (uniform_prob if arm in valid_arms else 0.0) for arm in ARM_ORDER}

    # Sample uniformly
    chosen_arm = np.random.choice(valid_arms)

    return chosen_arm, arm_probs


def log_decision(
    user_id: str,
    context: UserContext,
    chosen_arm: str,
    arm_probs: Dict[str, float],
    prediction_time_ms: float,
    fallback_used: bool,
):
    """
    Log prediction decision for online learning.

    Writes to logs/inference/decisions_{date}.jsonl
    """
    log_entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'user_id': user_id,
        'current_multiplier': context.current_effectivelevelmultiplier,
        'chosen_arm': chosen_arm,
        'arm_probabilities': arm_probs,
        'features': context.features,
        'prediction_time_ms': prediction_time_ms,
        'fallback_used': fallback_used,
    }

    decisions_file = LOGS_DIR / f"decisions_{datetime.now().strftime('%Y%m%d')}.jsonl"
    with open(decisions_file, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')


# ============================================================================
# API Endpoints
# ============================================================================

@app.post("/predict", response_model=PredictionResponse)
async def predict(context: UserContext):
    """
    Get difficulty recommendation for a user.

    Returns:
        Prediction with chosen arm and probabilities
    """
    REQUESTS_TOTAL.inc()
    ACTIVE_REQUESTS.inc()

    start_time = time.perf_counter()
    fallback_used = False

    try:
        # Extract features
        feature_vector = extract_feature_vector(context)

        # Feasible arms based on current multiplier
        feasible_arms = get_valid_arms(context.current_effectivelevelmultiplier)

        # Try VW prediction (probabilities)
        try:
            vw_example = format_vw_example(context, feature_vector)
            chosen_arm, arm_probs = predict_with_vw(vw_example, feasible_arms=feasible_arms)
        except Exception as e:
            logger.warning(f"VW prediction failed, using fallback: {e}")
            chosen_arm, arm_probs = predict_fallback(context)
            fallback_used = True
            FALLBACK_PREDICTIONS.inc()

        # Compute next multiplier
        next_mult = compute_next_multiplier(context.current_effectivelevelmultiplier, chosen_arm)

        # Prediction time
        prediction_time_ms = (time.perf_counter() - start_time) * 1000
        PREDICTION_LATENCY.observe(time.perf_counter() - start_time)

        # Log decision
        log_decision(
            context.user_id, context, chosen_arm, arm_probs,
            prediction_time_ms, fallback_used
        )

        return PredictionResponse(
            user_id=context.user_id,
            chosen_arm=chosen_arm,
            arm_probabilities=arm_probs,
            next_multiplier=next_mult,
            current_multiplier=context.current_effectivelevelmultiplier,
            feasible_arms=feasible_arms,
            prediction_time_ms=prediction_time_ms,
            model_version="vw_dr_v1",
            fallback_used=fallback_used
        )

    except Exception as e:
        REQUESTS_FAILED.inc()
        logger.error(f"Prediction error for user {context.user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        ACTIVE_REQUESTS.dec()


@app.get("/health")
async def health():
    """Health check endpoint."""
    endpoint_configured = bool(DATABRICKS_ENDPOINT_URL)
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "databricks_endpoint_configured": endpoint_configured,
        "databricks_endpoint_url": DATABRICKS_ENDPOINT_URL if endpoint_configured else None,
        "features_loaded": selected_features is not None,
        "propensity_loaded": propensity_model is not None,
        "fallback_available": vw_model_path is not None and vw_model_path.exists(),
    }


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "VW Contextual Bandit API",
        "version": "1.0.0",
        "endpoints": {
            "predict": "POST /predict",
            "health": "GET /health",
            "metrics": "GET /metrics"
        }
    }


if __name__ == "__main__":
    import uvicorn

    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )
