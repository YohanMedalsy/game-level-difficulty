#!/usr/bin/env python3
"""
Calibrate DR Scale Factor

After running Phase 1-4, this script calibrates the scale_factor used in Optuna's
DR estimation to match the actual OPE DR estimate from Phase 4 validation.

Usage:
    python scripts/calibrate_dr_scale.py

This will:
1. Load the best VW model from Phase 3
2. Compute VW's progressive validation loss
3. Load the actual OPE DR estimate from Phase 4
4. Calculate the optimal scale_factor
5. Update estimate_validation_dr_simple() in 03_train_vw_optuna.py

Or run manually to see the calculation.
"""

import sys
from pathlib import Path
import json
import re

# Add parent directory to path
BANDIT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(BANDIT_ROOT / "src"))

# Import from the actual script file (03_train_vw_optuna.py)
from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location("train_vw_optuna", BANDIT_ROOT / "scripts" / "03_train_vw_optuna.py")
train_module = module_from_spec(spec)
spec.loader.exec_module(train_module)
compute_progressive_validation_loss = train_module.compute_progressive_validation_loss
VALID_VW = train_module.VALID_VW

# Also import paths
DATA_PROCESSED = BANDIT_ROOT / "data" / "processed"

# Paths
MODELS_DIR = BANDIT_ROOT / "models"
ARTIFACTS_DIR = BANDIT_ROOT / "artifacts"
VALIDATION_REPORT = ARTIFACTS_DIR / "validation_report.md"


def get_vw_loss() -> float:
    """Get VW's progressive validation loss from best model."""
    model_path = MODELS_DIR / "vw_bandit_dr_best.vw"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Best model not found: {model_path}. Run Phase 3 first.")
    
    if not VALID_VW.exists():
        raise FileNotFoundError(f"Validation VW file not found: {VALID_VW}. Run Phase 2 first.")
    
    loss = compute_progressive_validation_loss(model_path, VALID_VW)
    return loss


def get_ope_dr_estimate() -> float:
    """Extract OPE DR estimate from Phase 4 validation report."""
    if not VALIDATION_REPORT.exists():
        raise FileNotFoundError(
            f"Validation report not found: {VALIDATION_REPORT}. Run Phase 4 first."
        )
    
    with open(VALIDATION_REPORT, 'r') as f:
        content = f.read()
    
    # Look for DR estimate pattern
    # Pattern: "DR: 1234.56 coins/day" or "**Test Set DR Estimate:** 1234.56 coins/day"
    patterns = [
        r'\*\*Test Set DR Estimate:\*\*\s*([\d.]+)',
        r'DR:\s*([\d.]+)\s*coins/day',
        r'DR estimate[:\s]+([\d.]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, content, re.IGNORECASE)
        if match:
            return float(match.group(1))
    
    # Alternative: try to parse from comparison CSV
    comparison_file = ARTIFACTS_DIR / "vw_vs_ope_comparison.csv"
    if comparison_file.exists():
        import pandas as pd
        df = pd.read_csv(comparison_file)
        if 'DR' in df.columns or 'DR Estimate' in df.columns:
            col = 'DR' if 'DR' in df.columns else 'DR Estimate'
            return float(df[col].iloc[0])
    
    raise ValueError(
        f"Could not extract DR estimate from validation report. "
        f"Please check {VALIDATION_REPORT} or provide DR estimate manually."
    )


def calculate_scale_factor(vw_loss: float, ope_dr: float) -> float:
    """
    Calculate optimal scale factor.
    
    Formula: scale_factor = ope_dr / (-vw_loss)
    
    Why?
    - VW loss is in cost space (lower is better)
    - OPE DR is in reward space (higher is better)
    - Conversion: reward = -cost * scale_factor
    - Solving: ope_dr = -vw_loss * scale_factor
    - Therefore: scale_factor = ope_dr / (-vw_loss)
    """
    if vw_loss >= 0:
        raise ValueError(
            f"VW loss should be negative (cost), got {vw_loss}. "
            f"Check if model was trained correctly."
        )
    
    scale_factor = ope_dr / (-vw_loss)
    return scale_factor


def update_optuna_script(scale_factor: float):
    """Update scale_factor in 03_train_vw_optuna.py."""
    script_path = BANDIT_ROOT / "scripts" / "03_train_vw_optuna.py"
    
    if not script_path.exists():
        print(f"⚠️  Script not found: {script_path}")
        return False
    
    with open(script_path, 'r') as f:
        content = f.read()
    
    # Find and replace scale_factor
    pattern = r'scale_factor\s*=\s*\d+'
    replacement = f'scale_factor = {scale_factor:.0f}'
    
    if re.search(pattern, content):
        new_content = re.sub(pattern, replacement, content)
        
        with open(script_path, 'w') as f:
            f.write(new_content)
        
        print(f"✅ Updated {script_path}")
        print(f"   Changed scale_factor to {scale_factor:.0f}")
        return True
    else:
        print(f"⚠️  Could not find scale_factor in {script_path}")
        return False


def main():
    """Main calibration function."""
    print("=" * 80)
    print("DR SCALE FACTOR CALIBRATION")
    print("=" * 80)
    print()
    
    try:
        # Step 1: Get VW loss
        print("Step 1: Getting VW progressive validation loss...")
        vw_loss = get_vw_loss()
        print(f"   ✅ VW loss: {vw_loss:.6f}")
        print()
        
        # Step 2: Get OPE DR estimate
        print("Step 2: Extracting OPE DR estimate from Phase 4 validation...")
        ope_dr = get_ope_dr_estimate()
        print(f"   ✅ OPE DR estimate: {ope_dr:.2f} coins/day")
        print()
        
        # Step 3: Calculate scale factor
        print("Step 3: Calculating optimal scale_factor...")
        scale_factor = calculate_scale_factor(vw_loss, ope_dr)
        print(f"   ✅ Optimal scale_factor: {scale_factor:.2f}")
        print()
        
        # Step 4: Verify calculation
        print("Verification:")
        estimated_dr = -vw_loss * scale_factor
        print(f"   VW loss * scale_factor = {estimated_dr:.2f} coins/day")
        print(f"   OPE DR estimate        = {ope_dr:.2f} coins/day")
        print(f"   Difference            = {abs(estimated_dr - ope_dr):.2f} coins/day")
        print()
        
        if abs(estimated_dr - ope_dr) < 10:
            print("   ✅ Calibration successful! Estimates align.")
        else:
            print(f"   ⚠️  Difference > 10 coins/day. Check calculation.")
        print()
        
        # Step 5: Update script
        print("Step 4: Updating 03_train_vw_optuna.py...")
        if update_optuna_script(scale_factor):
            print()
            print("✅ Calibration complete!")
            print()
            print(f"Next Optuna runs will use scale_factor = {scale_factor:.0f}")
            print(f"This ensures Optuna's DR estimates match Phase 4 OPE validation.")
        else:
            print()
            print("⚠️  Please manually update scale_factor in:")
            print(f"   bandit/scripts/03_train_vw_optuna.py")
            print(f"   Change: scale_factor = {scale_factor:.0f}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nManual calibration:")
        print("  1. Get VW loss from Phase 3 training logs")
        print("  2. Get OPE DR from Phase 4 validation report")
        print("  3. Calculate: scale_factor = ope_dr / (-vw_loss)")
        print("  4. Update scale_factor in estimate_validation_dr_simple()")
        sys.exit(1)


if __name__ == "__main__":
    main()

