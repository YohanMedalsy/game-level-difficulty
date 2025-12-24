#!/usr/bin/env python3
"""
REFERENCE ONLY: FastAPI Model Serving Example

This is a learning example showing how to wrap a VW model in FastAPI.
This is NOT used in the production pipeline.

For production:
- Phase 5 Batch: Uses direct VW loading (05_run_fastapi_batch.py)
- Serving API: Uses Redis lookup (05_serve_redis_lookup.py)

Usage (for learning):
  python scripts/fastapi_model_service_example.py \
    --model-path dbfs:/mnt/vw_pipeline/models_aug15/vw_bandit_dr_best.vw \
    --selected-features dbfs:/mnt/artifacts/selected_features_50.json \
    --port 8000
"""

import os
import sys
from pathlib import Path
from typing import Dict, List
import json

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# Add parent to path
BANDIT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(BANDIT_ROOT))

from src.constants import (
    ACTION_TO_DELTA,
    ARM_ORDER,
    get_valid_arms,
    compute_next_multiplier,
    assign_feature_to_namespace,
)

try:
    import vowpalwabbit as vw
    import numpy as np
    VW_AVAILABLE = True
except ImportError:
    VW_AVAILABLE = False

app = FastAPI(
    title="VW Model Serving Example",
    description="Reference implementation of VW model wrapped in FastAPI",
    version="1.0.0"
)

# Global state
vw_model = None
selected_features = None
namespace_features = None


class UserContext(BaseModel):
    """User context for prediction."""
    user_id: str
    current_effectivelevelmultiplier: float
    features: Dict[str, float]


class PredictionResponse(BaseModel):
    """Prediction response."""
    user_id: str
    chosen_arm: str
    arm_probabilities: Dict[str, float]
    next_multiplier: float
    current_multiplier: float
    action: float


@app.on_event("startup")
async def load_model():
    """Load VW model and features on startup."""
    global vw_model, selected_features, namespace_features
    
    # These would be set via environment variables or CLI args
    model_path = os.getenv("MODEL_PATH", "/dbfs/mnt/vw_pipeline/models_aug15/vw_bandit_dr_best.vw")
    features_path = os.getenv("SELECTED_FEATURES_PATH", "/dbfs/mnt/artifacts/selected_features_50.json")
    
    print(f"Loading model from: {model_path}")
    vw_model = vw.Workspace(f"-i {model_path} --quiet", enable_logging=False)
    
    print(f"Loading features from: {features_path}")
    with open(features_path, 'r') as f:
        data = json.load(f)
    selected_features = data['selected_features']
    
    # Organize into namespaces
    from collections import defaultdict
    ns_dict = defaultdict(list)
    for feature in selected_features:
        ns = assign_feature_to_namespace(feature)
        ns_dict[ns].append(feature)
    namespace_features = dict(ns_dict)
    
    print(f"✅ Model loaded with {len(selected_features)} features in {len(namespace_features)} namespaces")


@app.post("/predict", response_model=PredictionResponse)
async def predict(context: UserContext):
    """Make prediction for a user."""
    if vw_model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Get feasible arms
        feasible_arms = get_valid_arms(context.current_effectivelevelmultiplier)
        
        # Format VW example
        vw_example = format_vw_example(context, feasible_arms)
        
        # Make prediction
        predictions = vw_model.predict(vw_example)
        
        # Parse predictions
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
                probs_array = np.zeros(len(ARM_ORDER))
                for i, arm in enumerate(ARM_ORDER):
                    if arm in feasible_arms:
                        probs_array[i] = 1.0 / len(feasible_arms)
            
            arm_probs = {arm: float(probs_array[i]) for i, arm in enumerate(ARM_ORDER)}
            chosen_idx = int(np.random.choice(np.arange(len(ARM_ORDER)), p=probs_array))
            chosen_arm = ARM_ORDER[chosen_idx]
        else:
            # Fallback
            uniform_prob = 1.0 / len(feasible_arms)
            arm_probs = {arm: (uniform_prob if arm in feasible_arms else 0.0) for arm in ARM_ORDER}
            chosen_arm = np.random.choice(feasible_arms)
        
        # Calculate action and next multiplier
        action = ACTION_TO_DELTA[chosen_arm]
        next_mult = compute_next_multiplier(context.current_effectivelevelmultiplier, chosen_arm)
        
        return PredictionResponse(
            user_id=context.user_id,
            chosen_arm=chosen_arm,
            arm_probabilities=arm_probs,
            next_multiplier=next_mult,
            current_multiplier=context.current_effectivelevelmultiplier,
            action=action
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def format_vw_example(context: UserContext, feasible_arms: List[str]) -> str:
    """Format user context as VW CB_ADF example."""
    current_mult = context.current_effectivelevelmultiplier
    
    # Build shared context
    shared_parts = []
    for namespace, features in sorted(namespace_features.items()):
        ns_parts = []
        for feature in features:
            value = context.features.get(feature, 0.0)
            if value != 0:
                clean_name = feature.replace(' ', '_').replace(':', '_').replace('|', '_')
                ns_parts.append(f"{clean_name}:{value:.6f}")
        
        if ns_parts:
            shared_parts.append(f"|{namespace} " + " ".join(ns_parts))
    
    shared_line = "shared " + " ".join(shared_parts)
    
    # Build action lines
    action_lines = []
    for arm_idx, arm in enumerate(ARM_ORDER):
        delta = ACTION_TO_DELTA[arm]
        feasible = 1 if arm in feasible_arms else 0
        action_features = f"|a arm:{arm_idx} delta:{delta:.2f} |c mult:{current_mult:.2f} feasible:{feasible}"
        action_line = f"{arm_idx} {action_features}"
        action_lines.append(action_line)
    
    return shared_line + "\n" + "\n".join(action_lines)


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "healthy",
        "model_loaded": vw_model is not None,
        "features_loaded": selected_features is not None
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FastAPI Model Service Example (Reference)")
    parser.add_argument("--model-path", type=str, required=True, help="Path to VW model")
    parser.add_argument("--selected-features", type=str, required=True, help="Path to features JSON")
    parser.add_argument("--port", type=int, default=8000, help="Port to run on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()
    
    # Set environment variables for startup
    os.environ["MODEL_PATH"] = args.model_path
    os.environ["SELECTED_FEATURES_PATH"] = args.selected_features
    
    print("=" * 80)
    print("FASTAPI MODEL SERVICE EXAMPLE (REFERENCE ONLY)")
    print("=" * 80)
    print(f"Starting server on {args.host}:{args.port}")
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
