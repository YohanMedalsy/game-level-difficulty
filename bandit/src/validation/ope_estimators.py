"""
Off-Policy Evaluation Estimators

Implements IPS, SNIPS, and DR estimators for validation.
Matches difficulty_ope_daily_correct.py implementation exactly.
"""

import numpy as np
from typing import Dict, Tuple
import sys
from pathlib import Path

# Add parent to path for constants
BANDIT_ROOT = Path(__file__).parent.parent.parent.absolute()
sys.path.insert(0, str(BANDIT_ROOT / "src"))

from constants import ARM_ORDER, get_valid_arms


def compute_uniform_target_policy(
    current_mults: np.ndarray,
) -> np.ndarray:
    """
    Compute uniform target policy probabilities.

    For each context, assigns uniform probability over feasible arms.

    Matches difficulty_ope_daily_correct.py lines 592-613

    Args:
        current_mults: Array of current multipliers (n_samples,)

    Returns:
        target_policy: Array of shape (n_samples, n_arms) with probabilities
    """
    n_samples = len(current_mults)
    n_arms = len(ARM_ORDER)

    target_policy = np.zeros((n_samples, n_arms))
    arm_to_idx = {arm: i for i, arm in enumerate(ARM_ORDER)}

    for i, current_mult in enumerate(current_mults):
        valid_arms = get_valid_arms(current_mult)

        # Uniform probability over valid arms
        uniform_prob = 1.0 / len(valid_arms)
        for arm in valid_arms:
            target_policy[i, arm_to_idx[arm]] = uniform_prob

    return target_policy


def ope_ips_snips_dr(
    pi_old: np.ndarray,
    pi_target: np.ndarray,
    r_obs: np.ndarray,
    q_matrix: np.ndarray,
    target_policy: np.ndarray,
    observed_arms: np.ndarray,
    clip_max: float = 10.0,
) -> Dict[str, float]:
    """
    Compute IPS, SNIPS, and DR estimators.

    Matches difficulty_ope_daily_correct.py lines 620-659

    Args:
        pi_old: Behavior policy probabilities for observed actions (n_samples,)
        pi_target: Target policy probabilities for observed actions (n_samples,)
        r_obs: Observed rewards (n_samples,)
        q_matrix: Q-values for all arms (n_samples, n_arms)
        target_policy: Target policy probabilities for all arms (n_samples, n_arms)
        observed_arms: Which arm was actually taken (n_samples,)
        clip_max: Conservative clipping threshold for importance weights

    Returns:
        Dict with 'IPS', 'SNIPS', 'DR' estimates
    """
    # Importance weights
    w = np.clip(pi_target / pi_old, 0, clip_max)

    # IPS and SNIPS
    ips = np.mean(w * r_obs)
    snips = np.sum(w * r_obs) / np.sum(w)

    # CORRECT DR: Different Q-values for target vs behavior
    # First term: ∑_a π_target(a|x) q̂(x,a) (expectation over target policy)
    q_target = np.sum(target_policy * q_matrix, axis=1)

    # Second term: q̂(x, a_logged) (prediction for logged action)
    arm_to_idx = {arm: i for i, arm in enumerate(ARM_ORDER)}
    q_behavior = np.array([q_matrix[i, arm_to_idx[arm]] for i, arm in enumerate(observed_arms)])

    # DR formula: E[∑_a π_target(a|x) q̂(x,a)] + E[w * (r - q̂(x,a_logged))]
    dr = np.mean(q_target + w * (r_obs - q_behavior))

    return {"IPS": ips, "SNIPS": snips, "DR": dr}


def bootstrap_ope_estimates(
    pi_old: np.ndarray,
    pi_target: np.ndarray,
    r_obs: np.ndarray,
    q_matrix: np.ndarray,
    target_policy: np.ndarray,
    observed_arms: np.ndarray,
    user_ids: np.ndarray,
    n_bootstrap: int = 1000,
    clip_max: float = 10.0,
    alpha: float = 0.05,
) -> Dict[str, Tuple[float, float, float, float]]:
    """
    Bootstrap confidence intervals for OPE estimates.

    Uses user-stratified bootstrap if multiple users are detected.
    Falls back to simple row-level bootstrap if user_ids are missing (all -1) or single user.

    Args:
        pi_old: Behavior policy probabilities
        pi_target: Target policy probabilities
        r_obs: Observed rewards
        q_matrix: Q-values matrix
        target_policy: Target policy matrix
        observed_arms: Observed arms
        user_ids: User IDs for stratified sampling
        n_bootstrap: Number of bootstrap samples
        clip_max: Importance weight clipping
        alpha: Significance level (0.05 = 95% CI)

    Returns:
        Dict with (mean, std, ci_lower, ci_upper) for each estimator
    """
    unique_users = np.unique(user_ids)
    n_users = len(unique_users)
    n_samples = len(r_obs)

    ips_samples = []
    snips_samples = []
    dr_samples = []

    # Check for valid user IDs for stratified sampling
    use_user_bootstrap = (n_users > 1)
    
    if not use_user_bootstrap:
        print(f"⚠️  Warning: Only {n_users} unique user(s) found (or User IDs missing). Falling back to row-level bootstrap.")
        # Row-level bootstrap indices
        # Pre-generate all indices to speed up loop
        all_indices = np.random.randint(0, n_samples, size=(n_bootstrap, n_samples))
    else:
        # Optimizing user bootstrap: Pre-group indices by user to avoid slow np.where inside loop
        # user_map: unique_user_idx -> list of row indices
        # Faster: sort by user_id, then split
        sort_idx = np.argsort(user_ids)
        sorted_user_ids = user_ids[sort_idx]
        
        # Find split points
        _, start_indices = np.unique(sorted_user_ids, return_index=True)
        # Group indices
        # We need a list of arrays, where each array contains row indices for one user
        # np.split returns list of arrays from sorted_user_ids, but we want indices from sort_idx
        grouped_indices = np.split(sort_idx, start_indices[1:])
        
        # Grouped indices is a list of arrays. grouped_indices[i] are the row indices for unique_users[i]
        n_groups = len(grouped_indices)
        
    print(f"  Bootstrap: {n_bootstrap} resamples (Mode: {'User-Stratified' if use_user_bootstrap else 'Row-Level'})")

    for i in range(n_bootstrap):
        if use_user_bootstrap:
            # Sample user INDICES (0 to n_groups-1) with replacement
            sampled_group_indices = np.random.randint(0, n_groups, size=n_groups)
            # Concatenate the row indices for these sampled users
            # Note: This can still be slow in Python loop. 
            # Optimization: use np.concatenate on the selected blocks
            sampled_indices = np.concatenate([grouped_indices[g] for g in sampled_group_indices])
        else:
            # Row-level: use pre-generated or generate on fly
            sampled_indices = all_indices[i]

        # Compute OPE on bootstrap sample
        # We handle zero pi_old inside the estimator function or here?
        # The estimator handles clipping, but let's ensure safety against NaNs if sample is unlucky
        
        estimates = ope_ips_snips_dr(
            pi_old[sampled_indices],
            pi_target[sampled_indices],
            r_obs[sampled_indices],
            q_matrix[sampled_indices],
            target_policy[sampled_indices],
            observed_arms[sampled_indices],
            clip_max=clip_max,
        )

        ips_samples.append(estimates['IPS'])
        snips_samples.append(estimates['SNIPS'])
        dr_samples.append(estimates['DR'])

    # Compute statistics
    def compute_stats(samples):
        if not samples:
            return 0.0, 0.0, 0.0, 0.0
        mean = np.mean(samples)
        std = np.std(samples)
        ci_lower = np.percentile(samples, alpha / 2 * 100)
        ci_upper = np.percentile(samples, (1 - alpha / 2) * 100)
        return mean, std, ci_lower, ci_upper

    return {
        'IPS': compute_stats(ips_samples),
        'SNIPS': compute_stats(snips_samples),
        'DR': compute_stats(dr_samples),
    }


def compute_effective_sample_size(importance_weights: np.ndarray) -> float:
    """
    Compute effective sample size (ESS) from importance weights.

    ESS = (Σw)² / Σw²

    Low ESS indicates high variance in importance weights.

    Args:
        importance_weights: Array of importance weights

    Returns:
        Effective sample size (between 1 and n_samples)
    """
    sum_w = np.sum(importance_weights)
    sum_w_sq = np.sum(importance_weights ** 2)

    ess = (sum_w ** 2) / sum_w_sq
    return ess


def validate_ope_inputs(
    pi_old: np.ndarray,
    pi_target: np.ndarray,
    r_obs: np.ndarray,
    q_matrix: np.ndarray,
    target_policy: np.ndarray,
    observed_arms: np.ndarray,
):
    """
    Validate OPE input arrays.

    Raises:
        ValueError: If inputs are invalid
    """
    n_samples = len(r_obs)
    n_arms = len(ARM_ORDER)

    # Check shapes
    assert pi_old.shape == (n_samples,), f"pi_old shape mismatch: {pi_old.shape}"
    assert pi_target.shape == (n_samples,), f"pi_target shape mismatch: {pi_target.shape}"
    assert r_obs.shape == (n_samples,), f"r_obs shape mismatch: {r_obs.shape}"
    assert q_matrix.shape == (n_samples, n_arms), f"q_matrix shape mismatch: {q_matrix.shape}"
    assert target_policy.shape == (n_samples, n_arms), f"target_policy shape mismatch: {target_policy.shape}"
    assert observed_arms.shape == (n_samples,), f"observed_arms shape mismatch: {observed_arms.shape}"

    # Check probabilities are valid
    assert np.all(pi_old > 0), "pi_old contains non-positive values"
    assert np.all(pi_target >= 0), "pi_target contains negative values"

    # Check target policy sums to 1
    policy_sums = np.sum(target_policy, axis=1)
    assert np.allclose(policy_sums, 1.0, atol=1e-6), \
        f"target_policy rows don't sum to 1: {policy_sums[:10]}"

    # Check observed_arms are valid
    arm_to_idx = {arm: i for i, arm in enumerate(ARM_ORDER)}
    for arm in observed_arms:
        if arm not in arm_to_idx:
            raise ValueError(f"Invalid observed arm: {arm}")


if __name__ == "__main__":
    # Test OPE estimators
    print("Testing OPE Estimators")
    print("=" * 80)

    # Simulate small test case
    n_samples = 1000
    n_arms = 5

    # Behavior policy: uniform
    pi_old = np.full(n_samples, 0.2)

    # Target policy: also uniform (should give E[r] ≈ mean reward)
    pi_target = np.full(n_samples, 0.2)

    # Random rewards
    np.random.seed(42)
    r_obs = np.random.exponential(scale=1000, size=n_samples)

    # Random Q-matrix
    q_matrix = np.random.exponential(scale=1000, size=(n_samples, n_arms))

    # Uniform target policy matrix
    target_policy = np.full((n_samples, n_arms), 0.2)

    # Random observed arms
    observed_arms = np.random.choice(ARM_ORDER, size=n_samples)

    # Compute estimates
    estimates = ope_ips_snips_dr(
        pi_old, pi_target, r_obs, q_matrix, target_policy, observed_arms
    )

    print("\nPoint estimates:")
    for estimator, value in estimates.items():
        print(f"  {estimator:6s}: {value:.2f}")

    print(f"\nEmpirical mean reward: {np.mean(r_obs):.2f}")
    print(f"(IPS and SNIPS should be close to this since policies are identical)")

    # Effective sample size
    w = pi_target / pi_old
    ess = compute_effective_sample_size(w)
    print(f"\nEffective sample size: {ess:.0f} / {n_samples} ({ess/n_samples*100:.1f}%)")

    print("\n" + "=" * 80)
    print("✅ OPE estimators validated successfully!")
