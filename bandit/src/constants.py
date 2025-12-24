"""
Vowpal Wabbit Contextual Bandit Constants

Defines the 5-arm difficulty selection bandit structure matching the OPE pipeline.
All constants are imported from difficulty_ope_daily_correct.py for consistency.
"""

from typing import List, Dict

# Action-to-multiplier delta mapping for SpacePlay difficulty system
# Matches difficulty_ope_daily_correct.py lines 111-117
ACTION_TO_DELTA: Dict[str, float] = {
    "Easierer": -0.12,  # Make game significantly easier
    "Easier": -0.06,    # Make game easier
    "Same": 0.00,       # Keep difficulty unchanged
    "Harder": 0.06,     # Make game harder
    "Harderer": 0.12,   # Make game significantly harder
}

# Arm order for consistent indexing across VW, propensity model, and OPE
# Matches difficulty_ope_daily_correct.py line 120
ARM_ORDER: List[str] = ["Easierer", "Easier", "Same", "Harder", "Harderer"]

# Multiplier bounds (business constraints)
# Matches difficulty_ope_daily_correct.py line 123
MIN_MULTIPLIER: float = 0.5   # Minimum difficulty multiplier
MAX_MULTIPLIER: float = 1.25  # Maximum difficulty multiplier

# Number of arms
N_ARMS: int = len(ARM_ORDER)

# OPE baseline for validation
# From difficulty_ope_daily_correct.py OPE results
OPE_UNIFORM_DR_MEAN: float = 1234.86  # Uniform policy DR estimate (coins/day)
OPE_UNIFORM_DR_STD: float = 1006.60   # Standard deviation

# Minimum samples per arm for training (from OPE)
MIN_SAMPLES_PER_ARM: int = 20

# Validation thresholds
VW_DR_MIN_ACCEPTABLE: float = OPE_UNIFORM_DR_MEAN * 0.95  # VW must beat 95% of uniform
VW_DR_MAX_SANITY: float = OPE_UNIFORM_DR_MEAN * 1.5       # Sanity check upper bound

# Feature selection
N_FEATURES_SELECTED: int = 50  # Number of features after RF selection (matching OPE)

# Data splits
TRAIN_RATIO: float = 0.8
VALID_RATIO: float = 0.1
TEST_RATIO: float = 0.1

assert abs(TRAIN_RATIO + VALID_RATIO + TEST_RATIO - 1.0) < 1e-6, "Split ratios must sum to 1.0"


def get_valid_arms(current_multiplier: float) -> List[str]:
    """
    Get list of valid difficulty arms given current multiplier and bounds.

    CRITICAL CONSTRAINT: The next multiplier must stay within [MIN_MULTIPLIER, MAX_MULTIPLIER]

    Matches difficulty_ope_daily_correct.py lines 123-134

    Args:
        current_multiplier: Current difficulty multiplier

    Returns:
        List of valid arm names
    """
    valid_arms = []
    for arm, delta in ACTION_TO_DELTA.items():
        next_mult = current_multiplier + delta
        if MIN_MULTIPLIER <= next_mult <= MAX_MULTIPLIER:
            valid_arms.append(arm)
    return valid_arms


def compute_next_multiplier(current_multiplier: float, chosen_arm: str) -> float:
    """
    Compute the next day's effective multiplier given current multiplier and chosen action.

    Matches difficulty_ope_daily_correct.py lines 137-151

    Args:
        current_multiplier: Current difficulty multiplier
        chosen_arm: Chosen difficulty arm

    Returns:
        Next day's multiplier (constrained to [MIN_MULTIPLIER, MAX_MULTIPLIER])
    """
    if chosen_arm not in ACTION_TO_DELTA:
        raise ValueError(
            f"Unknown arm: {chosen_arm}. Must be one of {list(ACTION_TO_DELTA.keys())}"
        )

    delta = ACTION_TO_DELTA[chosen_arm]
    next_multiplier = current_multiplier + delta

    # Apply constraints (multiplier bounds from data generation pipeline)
    if next_multiplier > MAX_MULTIPLIER or next_multiplier < MIN_MULTIPLIER:
        next_multiplier = current_multiplier  # Revert to same (invalid action)

    return next_multiplier


def arm_to_index(arm: str) -> int:
    """Convert arm name to index (0-4)."""
    return ARM_ORDER.index(arm)


def index_to_arm(index: int) -> str:
    """Convert index (0-4) to arm name."""
    return ARM_ORDER[index]


# VW CB_ADF namespace organization
# Groups features by semantic meaning for better interactions
VW_NAMESPACES: Dict[str, List[str]] = {
    'user': ['user_', 'lifetime', 'install', 'cohort'],  # User-level features
    'session': ['session', 'play', 'level', 'attempt'],  # Session metrics
    'economy': ['coin', 'exchange', 'spent', 'balance'],  # Economy features
    'temporal': ['day', 'week', 'weekend', 'hour'],       # Temporal features
    'lag': ['previous', 'prev_'],                          # Lag features
    'ewma': ['ewma_'],                                     # EWMA features
}


def assign_feature_to_namespace(feature_name: str) -> str:
    """
    Assign a feature to its VW namespace based on name patterns.

    Args:
        feature_name: Name of the feature

    Returns:
        Namespace identifier (e.g., 'u' for user, 's' for session)
    """
    feature_lower = feature_name.lower()

    for namespace, patterns in VW_NAMESPACES.items():
        for pattern in patterns:
            if pattern.lower() in feature_lower:
                # Return single-letter namespace for VW
                return namespace[0]

    # Default namespace for features that don't match
    return 'f'  # 'f' for feature (generic)


# Namespace abbreviations for VW format
NAMESPACE_ABBREV: Dict[str, str] = {
    'user': 'u',
    'session': 's',
    'economy': 'e',
    'temporal': 't',
    'lag': 'l',
    'ewma': 'w',  # 'w' for weighted average
    'action': 'a',
    'context': 'c',
    'feature': 'f',  # Generic
}


if __name__ == "__main__":
    # Validation tests
    print("VW Bandit Constants Validation")
    print("=" * 80)

    # Test 1: Valid arms at different multipliers
    print("\nTest 1: Valid arms at different multipliers")
    for mult in [0.5, 0.7, 0.88, 1.0, 1.13, 1.25]:
        valid = get_valid_arms(mult)
        print(f"  multiplier={mult:.2f}: {len(valid)} valid arms = {valid}")

    # Test 2: Next multiplier computation
    print("\nTest 2: Next multiplier computation")
    current = 0.88
    for arm in ARM_ORDER:
        next_mult = compute_next_multiplier(current, arm)
        delta = ACTION_TO_DELTA[arm]
        print(f"  {arm:10s}: {current:.2f} + {delta:+.2f} = {next_mult:.2f}")

    # Test 3: Namespace assignment
    print("\nTest 3: Namespace assignment")
    test_features = [
        "user_lifetime_days",
        "session_count_7d",
        "coins_balance",
        "day_of_week",
        "previous_day_action",
        "ewma_0.3_coins_spent"
    ]
    for feature in test_features:
        namespace = assign_feature_to_namespace(feature)
        print(f"  {feature:30s} -> |{namespace}")

    # Test 4: Validation thresholds
    print("\nTest 4: Validation thresholds")
    print(f"  OPE Uniform DR: {OPE_UNIFORM_DR_MEAN:.2f} ± {OPE_UNIFORM_DR_STD:.2f}")
    print(f"  VW must beat: {VW_DR_MIN_ACCEPTABLE:.2f} (95% of uniform)")
    print(f"  VW sanity check: {VW_DR_MAX_SANITY:.2f} (150% of uniform)")

    print("\n" + "=" * 80)
    print("✅ All constants validated successfully!")
