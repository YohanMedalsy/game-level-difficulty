#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 0: Data Generation from Snowflake
========================================

Pure Spark version of the daily-level contextual bandit pipeline for SpacePlay
difficulty selection. This is Phase 0 of the VW contextual bandit pipeline,
which comes BEFORE Phase 1 (feature selection).

All feature engineering is implemented with Spark DataFrame operations and 
window functions (no pandas transformations).

What this script does:
1. Connects to Snowflake and queries raw game events
2. Performs comprehensive feature engineering (520+ features)
3. Computes next_day_reward (coins spent next day)
4. Assigns difficulty_arm labels
5. Outputs Delta/Parquet file for Phase 1

Outputs:
- daily_features_spark.delta (or .parquet) - Input for Phase 1

Pipeline Flow:
Phase 0 (this script): Snowflake → Feature Engineering → Delta File
Phase 1 (01_prepare_data_spark.py): Delta File → Feature Selection
Phase 2 (02_convert_delta_to_vw_spark.py): Delta File → VW Format
Phase 3 (03_train_vw_optuna.py): VW Format → Trained Model

Author: Codex (pure Spark translation)
Date: 2025-10-27
Updated: 2025-11-03 (Renamed to Phase 0 for bandit pipeline)
"""

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import os
import sys
import math
import re
import shutil
from typing import List, Tuple, Optional, Dict, Set
from datetime import datetime, timedelta

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import (
    StructType, StructField, DoubleType, IntegerType, LongType, StringType,
    FloatType, DecimalType, DateType, DataType
)
from pyspark import StorageLevel

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

# =============================================================================
# PIPELINE CONTROL CONFIG
# =============================================================================
# This config file allows controlling ALL pipeline parameters from Databricks
# without modifying Snowflake trigger. Snowflake sends --date and --snowflake-table.
# Edit this file in the repo, push to GitHub, pull to Databricks.
#
# Config file location: bandit/config/pipeline_params.json (in the repo)


def load_pipeline_config() -> dict:
    """
    Load pipeline control configuration from the repo.

    Searches for config/pipeline_params.json relative to this script,
    or in Databricks workspace paths.

    Returns empty dict if file not found or invalid.
    """
    # Build list of possible config paths
    config_paths = []

    # 1. Try relative to this script (works when running as python script)
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        repo_config = os.path.join(script_dir, "..", "config", "pipeline_params.json")
        config_paths.append(os.path.normpath(repo_config))
    except NameError:
        pass  # __file__ not defined (Databricks notebook)

    # 2. Databricks workspace paths (when pulled via git)
    config_paths.extend([
        "/Workspace/Users/yohan.medalsy@spaceplay.games/ai/bandit/config/pipeline_params.json",
        "/Workspace/Repos/yohan.medalsy@spaceplay.games/ai/bandit/config/pipeline_params.json",
    ])

    # Try each path
    for config_path in config_paths:
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                print(f"📋 Loaded pipeline config from: {config_path}")
                return config
            except (json.JSONDecodeError, IOError) as e:
                print(f"⚠️  Failed to load config from {config_path}: {e}")
                continue

    print(f"ℹ️  No pipeline config found. Using command-line arguments and defaults.")
    print(f"   Searched paths: {config_paths}")
    return {}


def apply_pipeline_config(args, config: dict) -> None:
    """
    Apply pipeline config to args. Command-line args take precedence.

    Priority:
    1. Command-line arguments (from Snowflake) - highest priority
    2. Pipeline config file (from Databricks) - default behavior
    3. Argparse defaults - lowest priority
    """
    if not config:
        return

    # Helper to check if an argument was explicitly passed on command line
    def was_arg_passed(cli_flag: str) -> bool:
        """Check if argument was explicitly passed on command line."""
        return any(arg == cli_flag or arg.startswith(cli_flag + '=') for arg in sys.argv)

    # Mapping: config_key -> (attr_name, cli_flag, is_bool_flag)
    config_mapping = {
        # Data source
        'snowflake_table': ('snowflake_table', '--snowflake-table', False),
        # Mode flags
        'inference': ('inference', '--inference', True),
        'use_sf_connector': ('use_sf_connector', '--use-sf-connector', True),
        # Output config
        'output_format': ('output_format', '--output-format', False),
        'partition_column': ('partition_column', '--partition-column', False),
        'out_dir': ('out_dir', '--out-dir', False),
        'write_mode': ('write_mode', '--write-mode', False),
        'checkpoint_dir': ('checkpoint_dir', '--checkpoint-dir', False),
        # Materialization flags
        'skip_advanced_materialization': ('skip_advanced_materialization', '--skip-advanced-materialization', True),
        'skip_final_materialization': ('skip_final_materialization', '--skip-final-materialization', True),
        'skip_checkpoint': ('skip_checkpoint', '--skip-checkpoint', True),
        # Partition config
        'final_write_partitions': ('final_write_partitions', '--final-write-partitions', False),
        'partitions_per_day': ('partitions_per_day', '--partitions-per-day', False),
        'shuffle_partitions': ('shuffle_partitions', '--shuffle-partitions', False),
        'write_partitions': ('write_partitions', '--write-partitions', False),
        'max_records_per_file': ('max_records_per_file', '--max-records-per-file', False),
        'local_dirs': ('local_dirs', '--local-dirs', False),
        # EWMA config
        'ewma_lookback_days': ('ewma_lookback_days', '--ewma-lookback-days', False),
        # Job chaining
        'trigger_phase5': ('trigger_phase5', '--trigger-phase5', True),
        'trigger_phase1': ('trigger_phase1', '--trigger-phase1', True),
        'skip_training': ('skip_training', '--skip-training', True),
        'phase5_job_id': ('phase5_job_id', '--phase5-job-id', False),
        'phase6_job_id': ('phase6_job_id', '--phase6-job-id', False),
        'phase1_job_id': ('phase1_job_id', '--phase1-job-id', False),
        'selected_features_path': ('selected_features_path', '--selected-features-path', False),
        'model_path': ('model_path', '--model-path', False),
        # Snowflake config
        'sf_config_json': ('sf_config_json', '--sf-config-json', False),
    }

    applied = []
    skipped = []

    for config_key, (attr_name, cli_flag, is_bool) in config_mapping.items():
        if config_key not in config:
            continue

        config_value = config[config_key]

        # Skip null values and comment keys
        if config_value is None or config_key.startswith('_'):
            continue

        # Check if this arg was explicitly passed on command line
        if was_arg_passed(cli_flag):
            skipped.append(f"{cli_flag} (CLI override)")
            continue

        # Apply the config value
        setattr(args, attr_name, config_value)
        if is_bool:
            applied.append(f"{cli_flag}")
        else:
            applied.append(f"{cli_flag}={config_value}")

    # Print config application summary
    print("\n" + "=" * 80)
    print("📋 PIPELINE CONFIG APPLIED")
    print("=" * 80)
    if config.get('reason'):
        print(f"   Reason: {config['reason']}")
    if config.get('last_updated'):
        print(f"   Last updated: {config['last_updated']}")

    if applied:
        print(f"\n   ✅ Applied from config ({len(applied)} settings):")
        for setting in applied:
            print(f"      • {setting}")

    if skipped:
        print(f"\n   ⏭️  CLI override (command-line takes precedence):")
        for setting in skipped:
            print(f"      • {setting}")

    print("\n   💡 To change settings:")
    print("      1. Edit bandit/config/pipeline_params.json in the repo")
    print("      2. Push to GitHub and pull to Databricks")
    print("=" * 80 + "\n")


# =============================================================================
# FEATURE OPTIMIZATION CONSTANTS (Option 2)
# =============================================================================
# These mappings define which features belong to which computation groups,
# enabling selective computation in inference mode.
# See: bandit/docs/FEATURE_OPTIMIZATION_PLAN.md

# Aggregation groups - features computed together in aggregate_daily_features()
# Each group corresponds to a separate aggregation query
AGGREGATION_GROUPS = {
    'basic': {
        # Core spend and activity metrics - ALWAYS required
        'features': [
            'current_spend', 'current_exchangespentamountcoins',
            'daily_spend_real_sum', 'daily_spend_real_mean', 'daily_spend_real_std', 
            'daily_spend_real_max', 'daily_spend_real_min',
            'daily_spend_coins_sum', 'daily_spend_coins_mean', 'daily_spend_coins_std',
            'daily_spend_coins_max', 'daily_spend_coins_min',
            'daily_spend_real', 'daily_spend_coins',
            'day_start_ts', 'day_end_ts',
            'user_segment', 'platform', 'user_level_start', 'user_level_end',
            'daily_event_count', 'daily_session_count'
        ],
        'required': True  # Always compute - needed for basic pipeline functionality
    },
    'level_perf': {
        # Level performance, FPS, boosters, resurrections, time metrics
        'features': [
            'levels_started', 'levels_completed', 'levels_failed',
            'total_level_attempts', 'avg_level_attempts', 'level_attempts_std',
            'level_attempts_max', 'level_attempts_min', 'distinct_levels_played',
            'resurrections_space_sum', 'resurrections_space_mean', 'resurrections_space_std', 'resurrections_space_max',
            'resurrections_time_sum', 'resurrections_time_mean', 'resurrections_time_std', 'resurrections_time_max',
            'resurrections_space', 'resurrections_time',
            'total_level_duration', 'avg_level_duration', 'level_duration_std', 
            'level_duration_max', 'level_duration_min',
            'avg_time_remaining', 'time_remaining_std', 'time_remaining_max', 'time_remaining_min',
            'undo_boosters_sum', 'undo_boosters_mean', 'undo_boosters_std', 'undo_boosters_max',
            'extrabox_boosters_sum', 'extrabox_boosters_mean', 'extrabox_boosters_std', 'extrabox_boosters_max',
            'hammer_boosters_sum', 'hammer_boosters_mean', 'hammer_boosters_std', 'hammer_boosters_max',
            'joker_boosters_sum', 'joker_boosters_mean', 'joker_boosters_std', 'joker_boosters_max',
            'prejoker_boosters_sum', 'prejoker_boosters_mean', 'prejoker_boosters_std', 'prejoker_boosters_max',
            'supersuit_boosters_sum', 'supersuit_boosters_mean', 'supersuit_boosters_std', 'supersuit_boosters_max',
            'undo_boosters', 'extrabox_boosters', 'hammer_boosters', 
            'joker_boosters', 'prejoker_boosters', 'supersuit_boosters',
            'avg_rank_easing', 'rank_easing_std', 'rank_easing_max', 'rank_easing_min',
            'avg_effective_multiplier', 'effective_multiplier_std', 'effective_multiplier_max', 'effective_multiplier_min',
            'avg_fps', 'fps_std', 'fps_max', 'fps_min'
        ],
        'required': False
    },
    'economy': {
        # Economy and balance features
        'features': [
            'days_since_install',
            'coin_balance_start', 'coin_balance_end',
            'life_balance_start', 'life_balance_end',
            'crown_count_start', 'crown_count_end',
            'undo_balance_start', 'undo_balance_end',
            'hammer_balance_start', 'hammer_balance_end',
            'extrabox_balance_start', 'extrabox_balance_end',
            'ad_events', 'ad_revenue'
        ],
        'required': False
    }
}

# Derived features computed after aggregation groups are joined
# Maps feature name -> list of base features it depends on
DERIVED_FEATURE_DEPS = {
    'current_avg_attemptperuser': ['total_level_attempts', 'distinct_levels_played'],
    'day_of_week': [],  # Computed from session_date
    'day_of_month': [],  # Computed from session_date
    'month': [],  # Computed from session_date
    'is_weekend': [],  # Computed from session_date
    'completion_rate': ['levels_completed', 'levels_started'],
    'total_boosters_used': ['undo_boosters', 'extrabox_boosters', 'hammer_boosters', 
                           'joker_boosters', 'prejoker_boosters', 'supersuit_boosters'],
    'booster_usage_rate': ['total_boosters_used', 'levels_started'],
    'total_resurrections': ['resurrections_space', 'resurrections_time'],
    'resurrection_rate': ['total_resurrections', 'levels_started'],
    'coin_balance_change': ['coin_balance_end', 'coin_balance_start'],
    'life_balance_change': ['life_balance_end', 'life_balance_start'],
    'user_level_gain': ['user_level_end', 'user_level_start'],
    'undo_balance_change': ['undo_balance_end', 'undo_balance_start'],
    'hammer_balance_change': ['hammer_balance_end', 'hammer_balance_start'],
    'extrabox_balance_change': ['extrabox_balance_end', 'extrabox_balance_start'],
    'daily_play_duration_seconds': ['day_end_ts', 'day_start_ts'],
    'avg_session_duration': ['daily_play_duration_seconds', 'daily_session_count'],
}

# Advanced features from add_advanced_gameplay_features_spark()
# Maps feature name -> list of features it depends on (can be base or other advanced)
ADVANCED_FEATURE_DEPS = {
    # Rolling window features (3-day)
    'rolling_win_rate_3d': ['levels_completed', 'levels_started'],
    'rolling_completions_3d': ['levels_completed'],
    'rolling_completion_rate_3d': ['levels_completed', 'levels_started'],
    
    # Rolling window features (5-day)
    'rolling_win_rate_5d': ['levels_completed', 'levels_started'],
    'rolling_completions_5d': ['levels_completed'],
    'rolling_completion_rate_5d': ['levels_completed', 'levels_started'],
    
    # Rolling window features (7-day)
    'rolling_win_rate_7d': ['levels_completed', 'levels_started'],
    
    # Streak features
    'consecutive_wins_streak': ['levels_completed'],
    'consecutive_losses_streak': ['levels_failed'],
    'consecutive_days_streak': [],  # Computed from session_date gaps
    
    # Difficulty/preference features
    'preferred_difficulty_3d': ['current_effectivelevelmultiplier'],
    'preferred_difficulty_5d': ['current_effectivelevelmultiplier'],
    'preferred_difficulty_7d': ['current_effectivelevelmultiplier'],
    'difficulty_variance_3d': ['current_effectivelevelmultiplier'],
    'difficulty_variance_5d': ['current_effectivelevelmultiplier'],
    'difficulty_variance_7d': ['current_effectivelevelmultiplier'],
    'difficulty_trend_3d': ['current_effectivelevelmultiplier'],
    'difficulty_change_magnitude_3d': ['current_effectivelevelmultiplier'],
    'time_at_current_difficulty': ['current_effectivelevelmultiplier'],
    'distance_from_comfort_zone': ['current_effectivelevelmultiplier', 'preferred_difficulty_7d'],
    
    # Skill variance features
    'skill_variance_3d': ['completion_rate'],
    'skill_variance_5d': ['completion_rate'],
    
    # Engagement features
    'high_engagement_days_3d': ['daily_play_duration_seconds'],
    'high_engagement_days_5d': ['daily_play_duration_seconds'],
    'session_intensity_3d': ['daily_session_count'],
    'playtime_trend_3d': ['daily_play_duration_seconds'],
    
    # Spending features
    'is_whale': ['current_spend'],
    'days_since_last_spend': ['current_spend'],
    'spending_frequency_7d': ['current_spend'],
    'avg_spend_per_active_day_7d': ['current_spend'],
    'spending_volatility_7d': ['current_spend'],
    'is_spender_7d': ['current_spend'],
    'days_since_last_play': [],  # Computed from session_date
    
    # Interaction features
    'skill_x_difficulty': ['completion_rate', 'avg_effective_multiplier'],
    'playtime_x_skill': ['daily_play_duration_seconds', 'completion_rate'],
    'spend_x_completion': ['current_spend', 'completion_rate'],
    
    # Complex derived features (depend on other advanced features)
    'recent_struggle_indicator': ['rolling_win_rate_3d', 'completion_rate'],
    'performance_momentum': ['rolling_completion_rate_3d'],
    'weekend_warrior': ['is_weekend', 'daily_play_duration_seconds'],
    
    # Lifecycle features
    'lifecycle_new_user': ['days_since_install'],
    'lifecycle_early': ['days_since_install'],
    'lifecycle_established': ['days_since_install'],
    'lifecycle_veteran': ['days_since_install'],
    'install_month': ['days_since_install'],
}


def parse_ewma_from_selected_features(features_path: str) -> Dict[str, Set[float]]:
    """
    Parse selected_features.json to extract which EWMA features are needed.
    
    EWMA feature naming pattern: ewma_{base_column}_alpha{N}
    where N = alpha * 10 (e.g., alpha3 means alpha=0.3)
    
    Args:
        features_path: Path to selected_features.json (local or DBFS)
        
    Returns:
        Dict mapping base column names to set of alpha values needed.
        e.g., {'fps_max': {0.1, 0.3, 0.5}, 'daily_spend_coins': {0.3}}
        
    Returns empty dict if file not found or parsing fails.
    """
    # Normalize DBFS path for local file reading
    if features_path.startswith("dbfs:/"):
        local_path = "/dbfs" + features_path[5:]
    else:
        local_path = features_path
    
    try:
        with open(local_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"⚠️  Could not load selected features from {local_path}: {e}")
        return {}
    
    features = data.get('selected_features', [])
    ewma_specs: Dict[str, Set[float]] = {}
    
    for feat in features:
        if feat.startswith('ewma_') and '_alpha' in feat:
            # Parse: ewma_{base_col}_alpha{N}
            # Use rsplit to handle base columns with underscores
            # e.g., ewma_daily_spend_coins_sum_alpha3 -> base=daily_spend_coins_sum, alpha=0.3
            try:
                parts = feat.rsplit('_alpha', 1)
                if len(parts) == 2:
                    base_col = parts[0][5:]  # Remove 'ewma_' prefix
                    alpha_int = int(parts[1])
                    alpha = alpha_int / 10.0
                    
                    if base_col not in ewma_specs:
                        ewma_specs[base_col] = set()
                    ewma_specs[base_col].add(alpha)
            except (ValueError, IndexError) as e:
                print(f"⚠️  Could not parse EWMA feature name '{feat}': {e}")
                continue
    
    if ewma_specs:
        total_ewma = sum(len(alphas) for alphas in ewma_specs.values())
        print(f"📊 INFERENCE MODE: Optimized EWMA computation")
        print(f"   Loaded {len(features)} selected features from {features_path}")
        print(f"   Found {total_ewma} EWMA features across {len(ewma_specs)} base columns")
        print(f"   (vs 680+ EWMA features in full computation)")
    
    return ewma_specs


def parse_required_features(features_path: str) -> Dict:
    """
    Parse selected_features.json and determine all required computations.
    
    This function analyzes which features are selected and traces back through
    the dependency graph to determine which aggregation groups and advanced
    features need to be computed.
    
    Args:
        features_path: Path to selected_features.json (local or DBFS)
        
    Returns:
        Dict with keys:
            'selected_features': List[str] - Original list from JSON
            'required_base': Set[str] - Base features to compute
            'required_derived': Set[str] - Derived features to compute
            'required_advanced': Set[str] - Advanced features to compute
            'required_groups': Set[str] - Aggregation groups needed
            'ewma_specs': Dict[str, Set[float]] - EWMA specs
    """
    # Normalize DBFS path for local file reading
    if features_path.startswith("dbfs:/"):
        local_path = "/dbfs" + features_path[5:]
    else:
        local_path = features_path
    
    result = {
        'selected_features': [],
        'required_base': set(),
        'required_derived': set(),
        'required_advanced': set(),
        'required_groups': set(),
        'ewma_specs': {}
    }
    
    try:
        with open(local_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"⚠️  Could not load selected features from {local_path}: {e}")
        return result
    
    selected_features = data.get('selected_features', [])
    result['selected_features'] = selected_features
    
    # Build lookup sets for each feature category
    all_base_features = set()
    for group_info in AGGREGATION_GROUPS.values():
        all_base_features.update(group_info['features'])
    
    all_derived_features = set(DERIVED_FEATURE_DEPS.keys())
    all_advanced_features = set(ADVANCED_FEATURE_DEPS.keys())
    
    # Helper function to recursively find dependencies
    def get_all_deps(feature: str, visited: Set[str] = None) -> Set[str]:
        """Recursively get all dependencies for a feature."""
        if visited is None:
            visited = set()
        if feature in visited:
            return set()
        visited.add(feature)
        
        deps = set()
        
        # Check derived features
        if feature in DERIVED_FEATURE_DEPS:
            for dep in DERIVED_FEATURE_DEPS[feature]:
                deps.add(dep)
                deps.update(get_all_deps(dep, visited))
        
        # Check advanced features
        if feature in ADVANCED_FEATURE_DEPS:
            for dep in ADVANCED_FEATURE_DEPS[feature]:
                deps.add(dep)
                deps.update(get_all_deps(dep, visited))
        
        return deps
    
    # Process each selected feature
    for feat in selected_features:
        feat_lower = feat.lower()
        
        # Check if it's an EWMA feature
        if feat_lower.startswith('ewma_') and '_alpha' in feat_lower:
            # Parse EWMA feature to get base column
            try:
                parts = feat_lower.rsplit('_alpha', 1)
                if len(parts) == 2:
                    base_col = parts[0][5:]  # Remove 'ewma_' prefix
                    alpha_int = int(parts[1])
                    alpha = alpha_int / 10.0
                    
                    if base_col not in result['ewma_specs']:
                        result['ewma_specs'][base_col] = set()
                    result['ewma_specs'][base_col].add(alpha)
                    
                    # The base column itself is a dependency
                    result['required_base'].add(base_col)
                    # Get all dependencies of the base column
                    result['required_base'].update(get_all_deps(base_col))
            except (ValueError, IndexError):
                pass
        
        # Check if it's an advanced feature
        elif feat_lower in {f.lower() for f in all_advanced_features}:
            # Find the actual feature name (case-insensitive match)
            for adv_feat in all_advanced_features:
                if adv_feat.lower() == feat_lower:
                    result['required_advanced'].add(adv_feat)
                    # Get dependencies
                    deps = get_all_deps(adv_feat)
                    for dep in deps:
                        if dep in all_advanced_features:
                            result['required_advanced'].add(dep)
                        elif dep in all_derived_features:
                            result['required_derived'].add(dep)
                        else:
                            result['required_base'].add(dep)
                    break
        
        # Check if it's a derived feature
        elif feat_lower in {f.lower() for f in all_derived_features}:
            for der_feat in all_derived_features:
                if der_feat.lower() == feat_lower:
                    result['required_derived'].add(der_feat)
                    # Get dependencies
                    deps = get_all_deps(der_feat)
                    for dep in deps:
                        if dep in all_advanced_features:
                            result['required_advanced'].add(dep)
                        elif dep in all_derived_features:
                            result['required_derived'].add(dep)
                        else:
                            result['required_base'].add(dep)
                    break
        
        # Otherwise it's a base feature
        else:
            result['required_base'].add(feat_lower)
    
    # Determine which aggregation groups are needed
    for group_name, group_info in AGGREGATION_GROUPS.items():
        # Always include required groups
        if group_info.get('required', False):
            result['required_groups'].add(group_name)
            continue
        
        # Check if any feature from this group is needed
        group_features_lower = {f.lower() for f in group_info['features']}
        required_base_lower = {f.lower() for f in result['required_base']}
        
        if group_features_lower & required_base_lower:
            result['required_groups'].add(group_name)
    
    # Print detailed summary
    print(f"\n" + "="*80)
    print(f"🔧 INFERENCE MODE: Feature Optimization Analysis")
    print(f"="*80)
    print(f"📄 Source: {features_path}")
    print(f"📊 Total selected features from JSON: {len(selected_features)}")
    print(f"-"*80)
    
    # Categorize selected features for display
    ewma_count = sum(len(v) for v in result['ewma_specs'].values())
    base_in_selected = len([f for f in selected_features if f.lower() in {b.lower() for b in result['required_base']}])
    adv_in_selected = len([f for f in selected_features if f.lower() in {a.lower() for a in result['required_advanced']}])
    
    print(f"\n📋 FEATURES TO BE COMPUTED:")
    print(f"   Base/Aggregation features: {len(result['required_base'])}")
    if result['required_base']:
        sorted_base = sorted(result['required_base'])
        print(f"      → {sorted_base[:10]}{'...' if len(sorted_base) > 10 else ''}")
    
    print(f"   Derived features: {len(result['required_derived'])}")
    if result['required_derived']:
        print(f"      → {sorted(result['required_derived'])}")
    
    print(f"   Advanced features (window funcs): {len(result['required_advanced'])}")
    if result['required_advanced']:
        print(f"      → {sorted(result['required_advanced'])}")
    
    print(f"   EWMA features: {ewma_count}")
    if result['ewma_specs']:
        for base_col, alphas in sorted(result['ewma_specs'].items()):
            alpha_strs = [f"alpha{int(a*10)}" for a in sorted(alphas)]
            print(f"      → ewma_{base_col}: {alpha_strs}")
    
    print(f"\n📦 AGGREGATION GROUPS TO COMPUTE: {result['required_groups']}")
    
    # Summary
    total_features_to_compute = len(result['required_base']) + len(result['required_derived']) + len(result['required_advanced']) + ewma_count
    print(f"\n✅ OPTIMIZATION SUMMARY:")
    print(f"   Features requested in JSON: {len(selected_features)}")
    print(f"   Total features to compute (incl. dependencies): {total_features_to_compute}")
    print(f"   EWMA base columns: {len(result['ewma_specs'])}")
    print(f"="*80 + "\n")
    
    return result


def initialize_spark_session(shuffle_partitions: int = 2000,
                             local_dirs: str = "",
                             enable_aqe: bool = True,
                             extra_jars: str = "",
                             extra_packages: str = "",
                             enable_delta_extensions: bool = False,
                             force_local_mode: bool = False) -> SparkSession:
    """Initialize Spark session; tune shuffles, AQE, and local dirs."""
    print("⚡ Initializing Spark session (Pure Spark pipeline)...")

    jar_path = "/Users/yohanmedalsy/Desktop/Personal_Projects/spark_jars/snowflake-jdbc-3.14.4.jar"
    if not os.path.exists(jar_path):
        jar_path = ""

    # Ensure driver and executor use the same Python version
    import sys
    python_executable = sys.executable
    
    is_databricks = bool(os.environ.get("DATABRICKS_RUNTIME_VERSION"))
    run_local = force_local_mode or not is_databricks
    if run_local:
        reason = "force_local_mode flag" if force_local_mode else "DATABRICKS_RUNTIME_VERSION not set"
        print(f"ℹ️  Running in local Spark mode ({reason}).")
    else:
        print("ℹ️  Detected Databricks runtime; using cluster Spark settings.")

    builder = SparkSession.builder.appName("SpacePlay_Daily_Bandit_Pipeline_PURE_SPARK")
    builder = builder.config("spark.sql.execution.arrow.pyspark.enabled", "true")
    builder = builder.config("spark.sql.shuffle.partitions", str(shuffle_partitions))
    builder = builder.config("spark.sql.adaptive.enabled", str(enable_aqe).lower())
    builder = builder.config("spark.sql.adaptive.coalescePartitions.enabled", str(enable_aqe).lower())
    builder = builder.config("spark.sql.adaptive.skewJoin.enabled", str(enable_aqe).lower())
    builder = builder.config("spark.sql.session.timeZone", "UTC")
    builder = builder.config("spark.databricks.delta.schema.autoMerge.enabled", "true")
    builder = builder.config("spark.databricks.delta.optimizeWrite.enabled", "false")
    # Disable Photon to prevent OOM on memory-constrained clusters
    # Photon is faster but uses significantly more memory, causing OOM with complex window functions
    builder = builder.config("spark.databricks.photon.enabled", "false")

    # Only override master/memory settings for local development
    if run_local:
        builder = (
            builder
            .config("spark.master", "local[*]")
            .config("spark.driver.memory", "64g")
            .config("spark.driver.maxResultSize", "32g")
            .config("spark.executor.memory", "32g")
            .config("spark.default.parallelism", str(shuffle_partitions))
        )

    builder = builder.config("spark.jars", jar_path)
    builder = builder.config("spark.pyspark.python", python_executable)
    builder = builder.config("spark.pyspark.driver.python", python_executable)

    if local_dirs:
        builder = builder.config("spark.local.dir", local_dirs)
        # Additional shuffle reliability configs when using custom local dirs
        # Spark 4.0: use spark.shuffle.localDisk.file.output.buffer instead of deprecated spark.shuffle.unsafe.file.output.buffer
        builder = builder \
            .config("spark.shuffle.localDisk.file.output.buffer", "512k") \
            .config("spark.shuffle.file.buffer", "512k") \
            .config("spark.shuffle.registration.timeout", "600000") \
            .config("spark.shuffle.registration.maxAttempts", "5")
    else:
        # When no custom local_dirs, ensure default temp directory stability
        builder = builder \
            .config("spark.shuffle.localDisk.file.output.buffer", "256k") \
            .config("spark.shuffle.file.buffer", "256k")
    if extra_jars:
        current_jars = jar_path if jar_path else ""
        combined = f"{current_jars},{extra_jars}" if current_jars and extra_jars else (extra_jars or current_jars)
        builder = builder.config("spark.jars", combined)
    if extra_packages:
        builder = builder.config("spark.jars.packages", extra_packages)
    if enable_delta_extensions:
        builder = builder \
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")

    spark = builder.getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    print("✅ Spark session initialized")
    print(f"   AQE: {enable_aqe}, shuffle partitions: {shuffle_partitions}")
    if local_dirs:
        print(f"   Local shuffle dirs: {local_dirs}")
    return spark


def connect_to_snowflake_direct() -> Tuple[dict, str]:
    """Return Snowflake connection parameters and JDBC URL."""
    print("🔗 Preparing Snowflake connection parameters...")

    snowflake_config = {
        "user": "YOUR_SNOWFLAKE_USER",
        "password": "YOUR_SNOWFLAKE_PASSWORD",
        "account": "YOUR_SNOWFLAKE_ACCOUNT",
        "warehouse": "ML_DATABRICKS",
        "database": "SPACEPLAY",
        "schema": "UNITY",
    }

    jdbc_url = f"jdbc:snowflake://{snowflake_config['account']}.snowflakecomputing.com/"

    print("✅ Snowflake configuration ready")
    return snowflake_config, jdbc_url


def load_snowflake_config_from_json(path: str) -> Optional[dict]:
    """Load Snowflake configuration from a JSON file if it exists.

    Expected keys (any subset is fine):
      url | account, user, password, warehouse, database, schema, role

    Returns None if the file is missing or malformed.
    """
    try:
        if not path:
            return None
        if not os.path.exists(path):
            print(f"ℹ️  Snowflake JSON config not found at: {path} (skipping)")
            return None
        with open(path, 'r') as f:
            data = json.load(f)
        if not isinstance(data, dict):
            print(f"⚠️  Snowflake JSON config at {path} is not an object; ignoring")
            return None
        # Normalize keys to upper/expected names
        norm = {k.lower(): v for k, v in data.items()}
        cfg = {
            'url': norm.get('url') or norm.get('sf_url') or norm.get('sfurl'),
            'account': norm.get('account') or norm.get('sf_account') or norm.get('sfaccount'),
            'user': norm.get('user') or norm.get('sf_user') or norm.get('sfuser'),
            'password': norm.get('password') or norm.get('sf_password') or norm.get('sfpassword'),
            'warehouse': norm.get('warehouse') or norm.get('sf_warehouse') or norm.get('sfwarehouse'),
            'database': norm.get('database') or norm.get('sf_database') or norm.get('sfdatabase'),
            'schema': norm.get('schema') or norm.get('sf_schema') or norm.get('sfschema'),
            'role': norm.get('role') or norm.get('sf_role') or norm.get('sfrole'),
        }
        return cfg
    except Exception as e:
        print(f"⚠️  Failed to load Snowflake JSON config from {path}: {e}")
        return None


def extract_date_from_table_name(table_name: str) -> Optional[str]:
    """
    Extract date from Snowflake table name.
    
    Examples:
        boxjam_daily_2025_10_20 -> 2025-10-20
        boxjam_daily_2025-10-20 -> 2025-10-20
        spaceplay.unity.boxjam_daily_2025_10_20 -> 2025-10-20
    
    Returns:
        Date string in YYYY-MM-DD format, or None if not found
    """
    import re
    # Look for pattern: _YYYY_MM_DD or _YYYY-MM-DD (handles both underscores and hyphens)
    date_match = re.search(r'_(\d{4})[_-](\d{2})[_-](\d{2})(?:\.|$)', table_name)
    if date_match:
        year, month, day = date_match.groups()
        return f"{year}-{month}-{day}"
    return None


def create_spark_dataframe(spark: SparkSession,
                           snowflake_config: dict,
                           jdbc_url: str,
                           start_date: str = "2025-07-01",
                           end_date: str = None,
                           snowflake_table: str = None) -> DataFrame:
    """Load raw SpacePlay events from Snowflake into Spark via JDBC."""
    print("📥 Loading SpacePlay events from Snowflake...")
    print(f"   Date filter: gauserstartdate >= '{start_date}'")
    if end_date:
        print(f"   Date filter: gauserstartdate <= '{end_date}'")
    
    # Use provided table name or default
    if snowflake_table:
        table_name = snowflake_table
    else:
        # Default table (backward compatible)
        table_name = "spaceplay.unity.boxjam_snapshot_2025_10_17"
    
    print(f"   Table: {table_name}")

    query = f"""
    SELECT *,
           MOD(ABS(HASH(USERID)::NUMBER(38,0)), 20) AS partition_key
    FROM {table_name}
    WHERE gameName = 'Box Jam'
      AND environmentName = 'Live'
      AND gauserstartdate >= '{start_date}'
    """
    if end_date:
        query += f" AND gauserstartdate <= '{end_date}'"

    print("   Using 20 partitions for parallel loading (hash partitioning)...")
    df = (
        spark.read
        .format("jdbc")
        .option("url", jdbc_url)
        .option("dbtable", f"({query}) AS subquery")
        .option("user", snowflake_config["user"])
        .option("password", snowflake_config["password"])
        .option("driver", "net.snowflake.client.jdbc.SnowflakeDriver")
        .option("partitionColumn", "partition_key")
        .option("lowerBound", "0")
        .option("upperBound", "19")
        .option("numPartitions", "20")
        .option("fetchsize", "10000")
        .load()
        .drop("partition_key")
    )

    print("✅ Snowflake data loaded (row count skipped for speed)")
    return df


def create_snowflake_connector_dataframe(spark: SparkSession,
                                         sf_options: dict,
                                         start_date: str = "2025-07-01",
                                         end_date: str = None,
                                         snowflake_table: str = None) -> DataFrame:
    """Load SpacePlay events using the Spark Snowflake Connector."""
    print("📥 Loading SpacePlay events via Spark Snowflake Connector...")
    print(f"   Date filter: gauserstartdate >= '{start_date}'")
    
    # Use provided table name or default
    if snowflake_table:
        table_name = snowflake_table
    else:
        # Default table (backward compatible)
        table_name = "spaceplay.unity.boxjam_snapshot_2025_10_17"
    
    print(f"   Table: {table_name}")

    query = f"""
    SELECT *
    FROM {table_name}
    WHERE gameName = 'Box Jam'
      AND environmentName = 'Live'
      AND gauserstartdate >= '{start_date}'
    """
    if end_date:
        query += f" AND gauserstartdate <= '{end_date}'"

    df = (
        spark.read
             .format("net.snowflake.spark.snowflake")
             .options(**sf_options)
             .option("query", query)
             .load()
    )
    print("✅ Snowflake Connector load complete (row count skipped)")
    return df


def load_from_parquet(spark: SparkSession, parquet_path: str) -> DataFrame:
    """Load raw data from Parquet export."""
    print(f"📂 Loading SpacePlay events from Parquet: {parquet_path}")
    df = spark.read.parquet(to_spark_path(parquet_path))
    if "partition_key" in df.columns:
        df = df.drop("partition_key")
    print("✅ Parquet data loaded (row count skipped for speed)")
    return df


def load_from_csv(spark: SparkSession, csv_path: str) -> DataFrame:
    """Load raw data from CSV export directory."""
    print(f"📂 Loading SpacePlay events from CSV: {csv_path}")
    df = spark.read.csv(csv_path, header=True, inferSchema=False)
    if "partition_key" in df.columns:
        df = df.drop("partition_key")
    print("✅ CSV data loaded (row count skipped for speed)")
    return df


def add_session_date_column(events_df: DataFrame, start_date: str = None, end_date: str = None) -> DataFrame:
    """Add session_date from timestamp columns and filter to requested range."""
    print("🗓️ Adding session_date column from EVENTTIMESTAMP/EVENTDATE...")
    
    if "EVENTTIMESTAMP" in events_df.columns:
        df = events_df.withColumn("session_date", F.to_date(F.col("EVENTTIMESTAMP")))
    elif "EVENTDATE" in events_df.columns:
        df = events_df.withColumn("session_date", F.col("EVENTDATE"))
    else:
        print("⚠️ No timestamp column found; defaulting to '2023-01-01'")
        df = events_df.withColumn("session_date", F.lit("2023-01-01"))

    # STRICT filtering by session_date to match requested range
    # gauserstartdate (used in SQL) is install date, so users can have events much later
    if start_date:
        print(f"   Filtering events to session_date >= {start_date}")
        df = df.filter(F.col("session_date") >= F.lit(start_date))
    
    if end_date:
        print(f"   Filtering events to session_date <= {end_date}")
        df = df.filter(F.col("session_date") <= F.lit(end_date))
        
    return df


def retain_multi_day_users(events_df: DataFrame,
                           user_col: str = "USERID",
                           date_col: str = "session_date") -> DataFrame:
    """Keep only users with activity on ≥2 distinct days."""
    print("🔍 Filtering to users with activity on ≥2 different days...")
    days_per_user = events_df.groupBy(user_col).agg(F.countDistinct(date_col).alias("distinct_session_days"))
    multi_day_users = days_per_user.filter(F.col("distinct_session_days") >= 2).select(user_col)
    filtered_df = events_df.join(multi_day_users, on=user_col, how="inner")
    print("✅ Multi-day user filtering complete (counts skipped)")
    return filtered_df


def extract_logged_action(daily_events: DataFrame) -> DataFrame:
    """Extract first LEVELEFFECTIVEMULTIPLIER from 'levelStarted' per user-day (assist=0)."""
    print("🎯 Extracting logged action from first levelStarted per user-day...")
    level_started = daily_events.filter(
        (F.col("EVENTNAME") == "levelStarted") &
        (F.col("LEVELEFFECTIVEMULTIPLIER").isNotNull()) &
        (F.col("LEVELDIFFICULTYDDAASSIST") == 0)
    )
    w = Window.partitionBy("USERID", "session_date").orderBy("EVENTTIMESTAMP")
    logged_action = (
        level_started
        .withColumn("row_num", F.row_number().over(w))
        .filter(F.col("row_num") == 1)
        .select(
            "USERID",
            "session_date",
            F.col("LEVELEFFECTIVEMULTIPLIER").alias("logged_difficulty_multiplier")
        )
    )
    print("✅ Logged action extracted")
    return logged_action


def assign_difficulty_arms(df_with_action: DataFrame, inference_mode: bool = False) -> DataFrame:
    """
    Assign 5 difficulty arms based on constrained action deltas (lagged next-day).
    
    Args:
        df_with_action: DataFrame with logged actions
        inference_mode: If True, keep rows without next-day data (for inference on target date)
    """
    print("🎲 Computing action deltas and assigning 5-arm difficulty labels...")
    w_user_day = Window.partitionBy("USERID").orderBy("session_date")

    df_with_next = df_with_action.withColumn(
        "next_day_multiplier_actual", F.lead("logged_difficulty_multiplier", 1).over(w_user_day)
    )

    if inference_mode:
        # INFERENCE MODE: Keep ALL rows, including those without next-day data (the target date)
        # For rows without next-day data, set action=0 (placeholder) and difficulty_arm="Inference"
        df_with_action_col = df_with_next.withColumn(
            "action",
            F.when(F.col("next_day_multiplier_actual").isNotNull(),
                   F.round(F.col("next_day_multiplier_actual") - F.col("logged_difficulty_multiplier"), 2))
             .otherwise(F.lit(0.00))  # Placeholder for inference rows
        )
        
        # One-sided directional constraints (bounds [0.5, 1.25]) - only for rows with actual next-day data
        df_with_action_col = df_with_action_col.withColumn(
            "action",
            F.when(F.col("next_day_multiplier_actual").isNull(), F.col("action"))  # Keep placeholder for inference
             .when((F.col("logged_difficulty_multiplier") <= 0.5) & (F.col("action") < 0), 0.00)
             .when((F.col("logged_difficulty_multiplier") >= 1.25) & (F.col("action") > 0), 0.00)
             .otherwise(F.col("action"))
        )
        
        df_with_action_col = df_with_action_col.withColumn(
            "next_effectivelevelmultiplier",
            F.when(F.col("next_day_multiplier_actual").isNotNull(),
                   F.round(F.col("logged_difficulty_multiplier") + F.col("action"), 2))
             .otherwise(F.col("logged_difficulty_multiplier"))  # Keep current for inference rows
        )
        
        df_with_arms = df_with_action_col.withColumn(
            "difficulty_arm",
            F.when(F.col("next_day_multiplier_actual").isNull(), "Inference")  # Special label for inference rows
             .when(F.col("action") == 0.00, "Same")
             .when((F.col("action") >= 0.05) & (F.col("action") <= 0.07), "Harder")
             .when((F.col("action") >= 0.11) & (F.col("action") <= 0.13), "Harderer")
             .when((F.col("action") >= -0.07) & (F.col("action") <= -0.05), "Easier")
             .when((F.col("action") >= -0.13) & (F.col("action") <= -0.11), "Easierer")
             .otherwise(None)
        )
        
        # In inference mode, keep rows with valid arm OR inference rows (target date)
        df_filtered = df_with_arms.filter(
            (F.col("difficulty_arm").isNotNull()) | (F.col("next_day_multiplier_actual").isNull())
        ).drop("next_day_multiplier_actual")
        print("✅ Arm assignment complete (inference mode: kept target date rows)")
    else:
        # TRAINING MODE: Original behavior - require next-day data
        df_with_action_col = df_with_next.withColumn(
            "action",
            F.round(F.col("next_day_multiplier_actual") - F.col("logged_difficulty_multiplier"), 2)
        ).filter(F.col("next_day_multiplier_actual").isNotNull())

        # One-sided directional constraints (bounds [0.5, 1.25])
        df_with_action_col = df_with_action_col.withColumn(
            "action",
            F.when((F.col("logged_difficulty_multiplier") <= 0.5) & (F.col("action") < 0), 0.00)
             .when((F.col("logged_difficulty_multiplier") >= 1.25) & (F.col("action") > 0), 0.00)
             .otherwise(F.col("action"))
        )

        df_with_action_col = df_with_action_col.withColumn(
            "next_effectivelevelmultiplier",
            F.round(F.col("logged_difficulty_multiplier") + F.col("action"), 2)
        )

        df_with_arms = df_with_action_col.withColumn(
            "difficulty_arm",
            F.when(F.col("action") == 0.00, "Same")
             .when((F.col("action") >= 0.05) & (F.col("action") <= 0.07), "Harder")
             .when((F.col("action") >= 0.11) & (F.col("action") <= 0.13), "Harderer")
             .when((F.col("action") >= -0.07) & (F.col("action") <= -0.05), "Easier")
             .when((F.col("action") >= -0.13) & (F.col("action") <= -0.11), "Easierer")
             .otherwise(None)
        )

        df_filtered = df_with_arms.filter(F.col("difficulty_arm").isNotNull()).drop("next_day_multiplier_actual")
        print("✅ Arm assignment complete")
    
    return df_filtered


def aggregate_daily_features(daily_events: DataFrame, 
                            required_features: Set[str] = None) -> DataFrame:
    """
    Aggregate raw events to one row per user-day with core features.
    
    Args:
        daily_events: Raw event DataFrame
        required_features: Optional set of feature names to compute. If None, compute all.
                          Used in inference mode to skip unnecessary derived features.
    """
    print("📊 Aggregating daily features (user-day level)...")
    
    # If required_features provided, build lowercase lookup for efficient matching
    required_features_lower = {f.lower() for f in required_features} if required_features else None

    # First event timestamp per user (for days_since_install)
    user_first_event = daily_events.groupBy("USERID").agg(
        F.min("EVENTTIMESTAMP").alias("first_event_timestamp")
    )
    daily_events = daily_events.join(user_first_event, on="USERID", how="left")

    # Basic daily aggregation
    daily_basic = daily_events.groupBy("USERID", "session_date").agg(
        F.sum(F.when(F.col("EXCHANGESPENTTYPE") == "RealCurrency", F.col("EXCHANGESPENTAMOUNT")).otherwise(0))
            .alias("current_spend"),
        F.sum(F.when(F.col("EXCHANGESPENTTYPE") == "VirtualCurrency", F.col("EXCHANGESPENTAMOUNT")).otherwise(0))
            .alias("current_exchangespentamountcoins"),

        # Real currency stats
        F.sum(F.when(F.col("EXCHANGESPENTTYPE") == "RealCurrency", F.col("EXCHANGESPENTAMOUNT")).otherwise(0))
            .alias("daily_spend_real_sum"),
        F.mean(F.when(F.col("EXCHANGESPENTTYPE") == "RealCurrency", F.col("EXCHANGESPENTAMOUNT")))
            .alias("daily_spend_real_mean"),
        F.stddev(F.when(F.col("EXCHANGESPENTTYPE") == "RealCurrency", F.col("EXCHANGESPENTAMOUNT")))
            .alias("daily_spend_real_std"),
        F.max(F.when(F.col("EXCHANGESPENTTYPE") == "RealCurrency", F.col("EXCHANGESPENTAMOUNT")))
            .alias("daily_spend_real_max"),
        F.min(F.when(F.col("EXCHANGESPENTTYPE") == "RealCurrency", F.col("EXCHANGESPENTAMOUNT")))
            .alias("daily_spend_real_min"),

        # Virtual currency stats
        F.sum(F.when(F.col("EXCHANGESPENTTYPE") == "VirtualCurrency", F.col("EXCHANGESPENTAMOUNT")).otherwise(0))
            .alias("daily_spend_coins_sum"),
        F.mean(F.when(F.col("EXCHANGESPENTTYPE") == "VirtualCurrency", F.col("EXCHANGESPENTAMOUNT")))
            .alias("daily_spend_coins_mean"),
        F.stddev(F.when(F.col("EXCHANGESPENTTYPE") == "VirtualCurrency", F.col("EXCHANGESPENTAMOUNT")))
            .alias("daily_spend_coins_std"),
        F.max(F.when(F.col("EXCHANGESPENTTYPE") == "VirtualCurrency", F.col("EXCHANGESPENTAMOUNT")))
            .alias("daily_spend_coins_max"),
        F.min(F.when(F.col("EXCHANGESPENTTYPE") == "VirtualCurrency", F.col("EXCHANGESPENTAMOUNT")))
            .alias("daily_spend_coins_min"),

        # Legacy compatibility
        F.sum(F.when(F.col("EXCHANGESPENTTYPE") == "RealCurrency", F.col("EXCHANGESPENTAMOUNT")).otherwise(0))
            .alias("daily_spend_real"),
        F.sum(F.when(F.col("EXCHANGESPENTTYPE") == "VirtualCurrency", F.col("EXCHANGESPENTAMOUNT")).otherwise(0))
            .alias("daily_spend_coins"),

        # Timestamps
        F.min("EVENTTIMESTAMP").alias("day_start_ts"),
        F.max("EVENTTIMESTAMP").alias("day_end_ts"),

        # User context (first/last of day)
        F.first("USERSKILLSEGMENT", ignorenulls=True).alias("user_segment"),
        F.first("PLATFORM", ignorenulls=True).alias("platform"),
        F.first("USERLEVEL", ignorenulls=True).alias("user_level_start"),
        F.last("USERLEVEL", ignorenulls=True).alias("user_level_end"),

        # Activity metrics
        F.count(F.lit(1)).alias("daily_event_count"),
        F.countDistinct("SESSIONID").alias("daily_session_count")
    )

    # Level performance metrics
    print("📊 Calculating daily level performance metrics...")
    level_events = daily_events.filter(
        F.col("EVENTNAME").isin(["levelStarted", "levelEnded", "levelOutOfSpace", "levelOutOfTime"]) 
    )
    level_features = level_events.groupBy("USERID", "session_date").agg(
        F.count(F.when(F.col("EVENTNAME") == "levelStarted", 1)).alias("levels_started"),
        F.count(F.when(
            F.col("EVENTNAME").isin(["levelEnded", "levelOutOfSpace", "levelOutOfTime"]) &
            (F.col("LEVELRESULT") == "Victory"), 1
        )).alias("levels_completed"),
        F.count(F.when(
            F.col("EVENTNAME").isin(["levelEnded", "levelOutOfSpace", "levelOutOfTime"]) &
            F.col("LEVELRESULT").isin(["Defeat", "Abort"]), 1
        )).alias("levels_failed"),

        # Attempts
        F.sum(F.when(F.col("LEVELATTEMPTS").isNotNull(), F.col("LEVELATTEMPTS")).otherwise(0)).alias("total_level_attempts"),
        F.mean(F.when(F.col("LEVELATTEMPTS").isNotNull(), F.col("LEVELATTEMPTS"))).alias("avg_level_attempts"),
        F.stddev(F.col("LEVELATTEMPTS")).alias("level_attempts_std"),
        F.max(F.col("LEVELATTEMPTS")).alias("level_attempts_max"),
        F.min(F.col("LEVELATTEMPTS")).alias("level_attempts_min"),

        F.countDistinct("LEVELINDEX").alias("distinct_levels_played"),

        # Resurrections (desperation signals - comprehensive statistics)
        F.sum(F.coalesce(F.col("LEVELOUTOFSPACERESURRECTIONS"), F.lit(0))).alias("resurrections_space_sum"),
        F.mean(F.col("LEVELOUTOFSPACERESURRECTIONS")).alias("resurrections_space_mean"),
        F.stddev(F.col("LEVELOUTOFSPACERESURRECTIONS")).alias("resurrections_space_std"),
        F.max(F.col("LEVELOUTOFSPACERESURRECTIONS")).alias("resurrections_space_max"),

        F.sum(F.coalesce(F.col("LEVELTIMEUPRESURRECTIONS"), F.lit(0))).alias("resurrections_time_sum"),
        F.mean(F.col("LEVELTIMEUPRESURRECTIONS")).alias("resurrections_time_mean"),
        F.stddev(F.col("LEVELTIMEUPRESURRECTIONS")).alias("resurrections_time_std"),
        F.max(F.col("LEVELTIMEUPRESURRECTIONS")).alias("resurrections_time_max"),

        # Legacy compatibility
        F.sum(F.coalesce(F.col("LEVELOUTOFSPACERESURRECTIONS"), F.lit(0))).alias("resurrections_space"),
        F.sum(F.coalesce(F.col("LEVELTIMEUPRESURRECTIONS"), F.lit(0))).alias("resurrections_time"),

        # Time metrics (comprehensive statistics)
        F.sum(F.coalesce(F.col("LEVELDURATION"), F.lit(0))).alias("total_level_duration"),
        F.mean(F.col("LEVELDURATION")).alias("avg_level_duration"),
        F.stddev(F.col("LEVELDURATION")).alias("level_duration_std"),
        F.max(F.col("LEVELDURATION")).alias("level_duration_max"),
        F.min(F.col("LEVELDURATION")).alias("level_duration_min"),

        F.mean(F.col("LEVELTIMEREMAINING")).alias("avg_time_remaining"),
        F.stddev(F.col("LEVELTIMEREMAINING")).alias("time_remaining_std"),
        F.max(F.col("LEVELTIMEREMAINING")).alias("time_remaining_max"),
        F.min(F.col("LEVELTIMEREMAINING")).alias("time_remaining_min"),

        # Booster usage (comprehensive statistics for each type)
        # Undo boosters
        F.sum(F.coalesce(F.col("LEVELUNDOBOOSTERUSED"), F.lit(0))).alias("undo_boosters_sum"),
        F.mean(F.col("LEVELUNDOBOOSTERUSED")).alias("undo_boosters_mean"),
        F.stddev(F.col("LEVELUNDOBOOSTERUSED")).alias("undo_boosters_std"),
        F.max(F.col("LEVELUNDOBOOSTERUSED")).alias("undo_boosters_max"),

        # Extrabox boosters
        F.sum(F.coalesce(F.col("LEVELEXTRABOXBOOSTERUSED"), F.lit(0))).alias("extrabox_boosters_sum"),
        F.mean(F.col("LEVELEXTRABOXBOOSTERUSED")).alias("extrabox_boosters_mean"),
        F.stddev(F.col("LEVELEXTRABOXBOOSTERUSED")).alias("extrabox_boosters_std"),
        F.max(F.col("LEVELEXTRABOXBOOSTERUSED")).alias("extrabox_boosters_max"),

        # Hammer boosters
        F.sum(F.coalesce(F.col("LEVELHAMMERBOOSTERUSED"), F.lit(0))).alias("hammer_boosters_sum"),
        F.mean(F.col("LEVELHAMMERBOOSTERUSED")).alias("hammer_boosters_mean"),
        F.stddev(F.col("LEVELHAMMERBOOSTERUSED")).alias("hammer_boosters_std"),
        F.max(F.col("LEVELHAMMERBOOSTERUSED")).alias("hammer_boosters_max"),

        # Joker boosters
        F.sum(F.coalesce(F.col("LEVELJOKERBOOSTERUSED"), F.lit(0))).alias("joker_boosters_sum"),
        F.mean(F.col("LEVELJOKERBOOSTERUSED")).alias("joker_boosters_mean"),
        F.stddev(F.col("LEVELJOKERBOOSTERUSED")).alias("joker_boosters_std"),
        F.max(F.col("LEVELJOKERBOOSTERUSED")).alias("joker_boosters_max"),

        # PreJoker boosters
        F.sum(F.coalesce(F.col("LEVELPREJOKERBOOSTERUSED"), F.lit(0))).alias("prejoker_boosters_sum"),
        F.mean(F.col("LEVELPREJOKERBOOSTERUSED")).alias("prejoker_boosters_mean"),
        F.stddev(F.col("LEVELPREJOKERBOOSTERUSED")).alias("prejoker_boosters_std"),
        F.max(F.col("LEVELPREJOKERBOOSTERUSED")).alias("prejoker_boosters_max"),

        # Supersuit boosters
        F.sum(F.coalesce(F.col("LEVELSUPERSUITBOOSTERUSED"), F.lit(0))).alias("supersuit_boosters_sum"),
        F.mean(F.col("LEVELSUPERSUITBOOSTERUSED")).alias("supersuit_boosters_mean"),
        F.stddev(F.col("LEVELSUPERSUITBOOSTERUSED")).alias("supersuit_boosters_std"),
        F.max(F.col("LEVELSUPERSUITBOOSTERUSED")).alias("supersuit_boosters_max"),

        # Legacy names for backward compatibility
        F.sum(F.coalesce(F.col("LEVELUNDOBOOSTERUSED"), F.lit(0))).alias("undo_boosters"),
        F.sum(F.coalesce(F.col("LEVELEXTRABOXBOOSTERUSED"), F.lit(0))).alias("extrabox_boosters"),
        F.sum(F.coalesce(F.col("LEVELHAMMERBOOSTERUSED"), F.lit(0))).alias("hammer_boosters"),
        F.sum(F.coalesce(F.col("LEVELJOKERBOOSTERUSED"), F.lit(0))).alias("joker_boosters"),
        F.sum(F.coalesce(F.col("LEVELPREJOKERBOOSTERUSED"), F.lit(0))).alias("prejoker_boosters"),
        F.sum(F.coalesce(F.col("LEVELSUPERSUITBOOSTERUSED"), F.lit(0))).alias("supersuit_boosters"),

        # Difficulty and performance (comprehensive statistics)
        F.mean(F.col("LEVELRANKEASING")).alias("avg_rank_easing"),
        F.stddev(F.col("LEVELRANKEASING")).alias("rank_easing_std"),
        F.max(F.col("LEVELRANKEASING")).alias("rank_easing_max"),
        F.min(F.col("LEVELRANKEASING")).alias("rank_easing_min"),

        F.mean(F.col("LEVELEFFECTIVEMULTIPLIER")).alias("avg_effective_multiplier"),
        F.stddev(F.col("LEVELEFFECTIVEMULTIPLIER")).alias("effective_multiplier_std"),
        F.max(F.col("LEVELEFFECTIVEMULTIPLIER")).alias("effective_multiplier_max"),
        F.min(F.col("LEVELEFFECTIVEMULTIPLIER")).alias("effective_multiplier_min"),

        F.mean(F.col("LEVELAVERAGEFPS")).alias("avg_fps"),
        F.stddev(F.col("LEVELAVERAGEFPS")).alias("fps_std"),
        F.max(F.col("LEVELAVERAGEFPS")).alias("fps_max"),
        F.min(F.col("LEVELAVERAGEFPS")).alias("fps_min")
    )

    # Economy & user state features
    print("📊 Adding daily economy and user state features...")
    economy_features = daily_events.groupBy("USERID", "session_date").agg(
        F.datediff(F.first("session_date"), F.first("first_event_timestamp")).alias("days_since_install"),

        F.first("USERCOINBALANCE", ignorenulls=True).alias("coin_balance_start"),
        F.last("USERCOINBALANCE", ignorenulls=True).alias("coin_balance_end"),

        F.first("USERLIFEBALANCE", ignorenulls=True).alias("life_balance_start"),
        F.last("USERLIFEBALANCE", ignorenulls=True).alias("life_balance_end"),

        F.first("USERCROWNCOUNT", ignorenulls=True).alias("crown_count_start")
            if "USERCROWNCOUNT" in daily_events.columns else F.lit(None).alias("crown_count_start"),
        F.last("USERCROWNCOUNT", ignorenulls=True).alias("crown_count_end")
            if "USERCROWNCOUNT" in daily_events.columns else F.lit(None).alias("crown_count_end"),

        F.first("USERUNDOBOOSTERBALANCE", ignorenulls=True).alias("undo_balance_start")
            if "USERUNDOBOOSTERBALANCE" in daily_events.columns else F.lit(0).alias("undo_balance_start"),
        F.last("USERUNDOBOOSTERBALANCE", ignorenulls=True).alias("undo_balance_end")
            if "USERUNDOBOOSTERBALANCE" in daily_events.columns else F.lit(0).alias("undo_balance_end"),
        F.first("USERHAMMERBOOSTERBALANCE", ignorenulls=True).alias("hammer_balance_start")
            if "USERHAMMERBOOSTERBALANCE" in daily_events.columns else F.lit(0).alias("hammer_balance_start"),
        F.last("USERHAMMERBOOSTERBALANCE", ignorenulls=True).alias("hammer_balance_end")
            if "USERHAMMERBOOSTERBALANCE" in daily_events.columns else F.lit(0).alias("hammer_balance_end"),

        F.first("USEREXTRABOXBOOSTERBALANCE", ignorenulls=True).alias("extrabox_balance_start")
            if "USEREXTRABOXBOOSTERBALANCE" in daily_events.columns else F.lit(0).alias("extrabox_balance_start"),
        F.last("USEREXTRABOXBOOSTERBALANCE", ignorenulls=True).alias("extrabox_balance_end")
            if "USEREXTRABOXBOOSTERBALANCE" in daily_events.columns else F.lit(0).alias("extrabox_balance_end"),

        F.count(F.when(F.col("EVENTNAME").like("%ad%") | F.col("EVENTNAME").like("%Ad%"), 1)).alias("ad_events"),
        F.sum(F.when(F.col("ADECPMUSD").isNotNull(), F.col("ADECPMUSD").cast("double")).otherwise(0.0)).alias("ad_revenue")
    )

    # Join
    print("📊 Joining all daily feature groups...")
    daily_aggregated = (
        daily_basic
        .join(level_features, on=["USERID", "session_date"], how="left")
        .join(economy_features, on=["USERID", "session_date"], how="left")
    )

    # Derived metrics - computed using simple column operations (fast)
    # Note: These are not conditionally skipped because they're very fast and many features depend on them
    print("📊 Computing derived metrics...")
    if required_features_lower:
        print(f"   (Inference mode: {len(required_features_lower)} features targeted)")
    
    # Helper to check if a feature or its dependents are needed
    def should_compute(feature_name: str) -> bool:
        if required_features_lower is None:
            return True  # Training mode: compute all
        # Check if this feature is directly needed OR if it's a dependency for other features
        return feature_name.lower() in required_features_lower
    
    # Always compute these core derived features (fast and widely used as dependencies)
    daily_aggregated = daily_aggregated \
        .withColumn(
            "current_avg_attemptperuser",
            F.when(F.col("distinct_levels_played") > 0,
                   F.col("total_level_attempts") / F.col("distinct_levels_played")).otherwise(0.0)
        ) \
        .withColumn("day_of_week", F.dayofweek(F.col("session_date"))) \
        .withColumn("day_of_month", F.dayofmonth(F.col("session_date"))) \
        .withColumn("month", F.month(F.col("session_date"))) \
        .withColumn("is_weekend", F.when(F.dayofweek(F.col("session_date")).isin([1, 7]), 1).otherwise(0)) \
        .withColumn(
            "completion_rate",
            F.when(F.col("levels_started") > 0,
                   F.col("levels_completed") / F.col("levels_started")).otherwise(0.0)
        ) \
        .withColumn(
            "total_boosters_used",
            F.coalesce(F.col("undo_boosters"), F.lit(0)) +
            F.coalesce(F.col("extrabox_boosters"), F.lit(0)) +
            F.coalesce(F.col("hammer_boosters"), F.lit(0)) +
            F.coalesce(F.col("joker_boosters"), F.lit(0)) +
            F.coalesce(F.col("prejoker_boosters"), F.lit(0)) +
            F.coalesce(F.col("supersuit_boosters"), F.lit(0))
        ) \
        .withColumn(
            "booster_usage_rate",
            F.when(F.col("levels_started") > 0,
                   F.col("total_boosters_used") / F.col("levels_started")).otherwise(0.0)
        ) \
        .withColumn(
            "total_resurrections",
            F.coalesce(F.col("resurrections_space"), F.lit(0)) +
            F.coalesce(F.col("resurrections_time"), F.lit(0))
        ) \
        .withColumn(
            "resurrection_rate",
            F.when(F.col("levels_started") > 0,
                   F.col("total_resurrections") / F.col("levels_started")).otherwise(0.0)
        ) \
        .withColumn("coin_balance_change", F.coalesce(F.col("coin_balance_end"), F.lit(0)) - F.coalesce(F.col("coin_balance_start"), F.lit(0))) \
        .withColumn("life_balance_change", F.coalesce(F.col("life_balance_end"), F.lit(0)) - F.coalesce(F.col("life_balance_start"), F.lit(0))) \
        .withColumn("user_level_gain", F.coalesce(F.col("user_level_end"), F.lit(0)) - F.coalesce(F.col("user_level_start"), F.lit(0))) \
        .withColumn("undo_balance_change", F.coalesce(F.col("undo_balance_end"), F.lit(0)) - F.coalesce(F.col("undo_balance_start"), F.lit(0))) \
        .withColumn("hammer_balance_change", F.coalesce(F.col("hammer_balance_end"), F.lit(0)) - F.coalesce(F.col("hammer_balance_start"), F.lit(0))) \
        .withColumn("extrabox_balance_change", F.coalesce(F.col("extrabox_balance_end"), F.lit(0)) - F.coalesce(F.col("extrabox_balance_start"), F.lit(0))) \
        .withColumn("daily_play_duration_seconds", F.unix_timestamp(F.col("day_end_ts")) - F.unix_timestamp(F.col("day_start_ts"))) \
        .withColumn("avg_session_duration", F.when(F.col("daily_session_count") > 0, F.col("daily_play_duration_seconds") / F.col("daily_session_count")).otherwise(0.0))

    # Fill some common nulls
    numeric_cols = [
        "current_spend", "current_exchangespentamountcoins", "levels_started",
        "levels_completed", "levels_failed", "total_boosters_used", "total_resurrections",
        "ad_events", "ad_revenue"
    ]
    daily_aggregated = daily_aggregated.fillna(0, subset=[c for c in numeric_cols if c in daily_aggregated.columns])

    print("✅ Daily feature aggregation complete")
    return daily_aggregated


def create_daily_bandit_dataset(daily_features: DataFrame,
                                logged_actions: DataFrame,
                                arm_assignments: DataFrame,
                                inference_mode: bool = False) -> DataFrame:
    """
    Join features + logged actions + assigned arms, add temporal targets/lag features.
    
    Args:
        inference_mode: If True, skip next_day_reward filter (for single-day inference data)
    """
    print("🔗 Creating daily bandit dataset...")
    df_with_action = daily_features.join(
        logged_actions.withColumnRenamed("logged_difficulty_multiplier", "current_effectivelevelmultiplier"),
        on=["USERID", "session_date"], how="inner"
    )
    df_with_arms = df_with_action.join(
        arm_assignments.select("USERID", "session_date", "action", "next_effectivelevelmultiplier", "difficulty_arm"),
        on=["USERID", "session_date"], how="inner"
    )

    w = Window.partitionBy("USERID").orderBy("session_date")
    df_with_temporal = df_with_arms.withColumn(
        "next_exchangespentamountcoins", F.lead("current_exchangespentamountcoins", 1).over(w)
    ).withColumn(
        "next_day_reward", F.col("next_exchangespentamountcoins")
    ).withColumn(
        "previous_day_action", F.lag("action", 1).over(w)
    ).withColumn(
        "previous_day_multiplier", F.lag("current_effectivelevelmultiplier", 1).over(w)
    )

    # For inference mode, don't filter out rows without next_day_reward (they won't exist for single-day data)
    if inference_mode:
        print("⚠️  Inference mode: Keeping all rows (next_day_reward will be NULL for single-day data)")
        df_final = df_with_temporal.withColumn(
            "previous_day_action", F.coalesce(F.col("previous_day_action"), F.lit(0.00))
        ).withColumn(
            "previous_day_multiplier", F.coalesce(F.col("previous_day_multiplier"), F.col("current_effectivelevelmultiplier"))
        )
        print("✅ Bandit dataset created (all rows kept for inference)")
    else:
        # Training mode: only keep rows with next_day_reward (need Day T+1 data)
        df_final = df_with_temporal.filter(F.col("next_exchangespentamountcoins").isNotNull()) \
            .withColumn("previous_day_action", F.coalesce(F.col("previous_day_action"), F.lit(0.00))) \
            .withColumn("previous_day_multiplier", F.coalesce(F.col("previous_day_multiplier"), F.col("current_effectivelevelmultiplier")))
        print("✅ Bandit dataset created (rows with next-day reward kept)")

    return df_final


def normalize_to_user_lowercase(df: DataFrame) -> DataFrame:
    """Normalize identifier column names for downstream consistency."""
    rename_map = {}
    if "USERID" in df.columns:
        rename_map["USERID"] = "user_id"
    # Spark doesn't support bulk rename; chain with select/alias
    cols = [F.col(c).alias(rename_map.get(c, c)) for c in df.columns]
    return df.select(*cols)


def clean_null_and_zero_columns_spark(df: DataFrame, id_cols: List[str]) -> DataFrame:
    """
    Drop columns that are:
    1) all NULL, or 2) numeric and ≥99% zeros (including all zeros).
    Implemented using Spark aggregations only.
    """
    print("🧹 Cleaning null and low-variance (zeros) columns in Spark...")

    # Quick empty check (cheaper than full count)
    if df.head(1) == []:
        return df

    # Determine numeric columns
    numeric_types = {"int", "bigint", "double", "float", "decimal", "smallint", "tinyint"}
    numeric_cols = [c for c, t in df.dtypes if any(nt in t for nt in numeric_types)]
    candidates = [c for c in df.columns if c not in id_cols]

    # Null counts + total count (compute in single pass)
    # NOTE: Using sample to avoid materializing entire DataFrame
    # Sample 1% of data for statistics (good enough for column selection)
    sample_df = df.sample(fraction=0.01, seed=42) if len(candidates) > 0 else df.limit(1)
    null_aggs = [F.sum(F.when(F.col(c).isNull(), 1).otherwise(0)).alias(f"{c}__nulls") for c in candidates]
    null_aggs.append(F.count(F.lit(1)).alias("__total_count"))
    null_row = sample_df.agg(*null_aggs).collect()[0].asDict()
    total_count = int(null_row.pop("__total_count", 0))
    # Scale up the sample count to estimate total
    total_count = total_count * 100  # Scale from 1% sample

    # Zero counts for numeric columns
    zero_aggs = [F.sum(F.when(F.col(c) == 0, 1).otherwise(0)).alias(f"{c}__zeros") for c in numeric_cols]
    zero_row = sample_df.agg(*zero_aggs).collect()[0].asDict() if zero_aggs else {}

    drop_cols = []
    for c in candidates:
        nulls = int(null_row.get(f"{c}__nulls", 0))
        if nulls == total_count:
            drop_cols.append(c)
            continue
        if c in numeric_cols:
            zeros = int(zero_row.get(f"{c}__zeros", 0))
            zero_pct = (zeros / total_count) * 100.0
            if zero_pct >= 99.0:
                drop_cols.append(c)

    if drop_cols:
        print(f"   Dropping {len(drop_cols)} columns: {', '.join(sorted(drop_cols)[:12])}{' ...' if len(drop_cols)>12 else ''}")
        df = df.drop(*drop_cols)
    else:
        print("   No columns dropped")
    return df


def add_advanced_gameplay_features_spark(df: DataFrame, 
                                         feature_whitelist: Set[str] = None) -> DataFrame:
    """
    Compute advanced gameplay features using Spark window functions only.
    
    Args:
        df: Input DataFrame with base features
        feature_whitelist: Optional set of feature names to compute. If None, compute all.
                          Used in inference mode to skip unnecessary expensive window functions.
    """
    print("\n" + "="*80)
    print("🎮 Computing advanced gameplay features (pure Spark)...")
    print("="*80)

    # We expect identifiers as user_id and session_date
    df = normalize_to_user_lowercase(df)
    original_cols = set(df.columns)
    
    # Build lowercase whitelist for case-insensitive matching
    whitelist_lower = {f.lower() for f in feature_whitelist} if feature_whitelist else None
    
    if feature_whitelist:
        print(f"📋 OPTIMIZATION MODE: Computing only {len(feature_whitelist)} whitelisted features")
        print(f"   Whitelisted features: {sorted(feature_whitelist)}")
    else:
        print(f"📋 FULL MODE: Computing ALL advanced features (training mode)")
    print("-"*80)
    
    # Helper function to check if a feature should be computed
    def should_compute(*feature_names) -> bool:
        """Return True if any of the feature_names should be computed."""
        if whitelist_lower is None:
            return True  # Training mode: compute all
        return any(f.lower() in whitelist_lower for f in feature_names)
    
    # Track which feature groups we're computing
    groups_computed = []
    groups_skipped = []
    features_computed = []
    features_skipped = []

    w = Window.partitionBy("user_id").orderBy("session_date")

    def lagged_avg(col: str, k: int):
        return F.avg(F.col(col)).over(w.rowsBetween(-k, -1))

    def lagged_sum(col: str, k: int):
        return F.sum(F.col(col)).over(w.rowsBetween(-k, -1))

    def lagged_std(col: str, k: int):
        return F.stddev_samp(F.col(col)).over(w.rowsBetween(-k, -1))

    # 1) Rolling win rate (completion_rate)
    rolling_win_features = [f"rolling_win_rate_{k}d" for k in [3, 5, 7]]
    if should_compute(*rolling_win_features, 'performance_momentum', 'recent_struggle_indicator'):
        groups_computed.append('rolling_win_rate')
        features_computed.extend(rolling_win_features)
        print(f"   ✅ Computing: rolling_win_rate → {rolling_win_features}")
        for k in [3, 5, 7]:
            df = df.withColumn(f"rolling_win_rate_{k}d", lagged_avg("completion_rate", k))
    else:
        groups_skipped.append('rolling_win_rate')
        features_skipped.extend(rolling_win_features)

    # 2) Difficulty comfort zone
    difficulty_features = [f"difficulty_variance_{k}d" for k in [3, 5, 7]] + [f"preferred_difficulty_{k}d" for k in [3, 5, 7]] + ['distance_from_comfort_zone']
    if should_compute(*difficulty_features):
        groups_computed.append('difficulty_comfort')
        features_computed.extend(difficulty_features)
        print(f"   ✅ Computing: difficulty_comfort → {difficulty_features}")
        for k in [3, 5, 7]:
            df = df.withColumn(f"difficulty_variance_{k}d", lagged_std("current_effectivelevelmultiplier", k)) \
                   .withColumn(f"preferred_difficulty_{k}d", lagged_avg("current_effectivelevelmultiplier", k))
    else:
        groups_skipped.append('difficulty_comfort')
        features_skipped.extend(difficulty_features)

    # 3) Consecutive days played streak (lagged)
    # Need prev_date for multiple features, always compute
    prev_date = F.lag("session_date", 1).over(w)
    
    if should_compute('consecutive_days_streak'):
        groups_computed.append('days_streak')
        features_computed.append('consecutive_days_streak')
        print(f"   ✅ Computing: days_streak → ['consecutive_days_streak']")
        is_new_segment = F.when(prev_date.isNull() | (F.datediff(F.col("session_date"), prev_date) > 1), 1).otherwise(0)
        seg_id = F.sum(is_new_segment).over(w.rowsBetween(Window.unboundedPreceding, 0))
        df = df.withColumn("__seg_id_days__", seg_id)
        w_seg = Window.partitionBy("user_id", "__seg_id_days__").orderBy("session_date")
        streak_incl = F.row_number().over(w_seg)
        df = df.withColumn("__streak_days_curr__", streak_incl)
        df = df.withColumn("consecutive_days_streak", F.coalesce(F.lag("__streak_days_curr__", 1).over(w), F.lit(0)))
    else:
        groups_skipped.append('days_streak')
        features_skipped.append('consecutive_days_streak')

    # 4) Performance momentum: recent (last 3) - older (days -10..-4)
    if should_compute('performance_momentum'):
        groups_computed.append('performance_momentum')
        features_computed.append('performance_momentum')
        print(f"   ✅ Computing: performance_momentum → ['performance_momentum']")
        recent_perf = lagged_avg("completion_rate", 3)
        older_perf = F.avg(F.col("completion_rate")).over(w.rowsBetween(-10, -4))
        df = df.withColumn("performance_momentum", recent_perf - older_perf)
    else:
        groups_skipped.append('performance_momentum')
        features_skipped.append('performance_momentum')

    # 5) Rolling completion stats and rates
    rolling_completion_features = [f"rolling_completions_{k}d" for k in [3, 5]] + [f"rolling_completion_rate_{k}d" for k in [3, 5]]
    if should_compute(*rolling_completion_features, 'win_loss_ratio_3d'):
        groups_computed.append('rolling_completions')
        features_computed.extend(rolling_completion_features)
        print(f"   ✅ Computing: rolling_completions → {rolling_completion_features}")
        for k in [3, 5]:
            comp = lagged_sum("levels_completed", k)
            starts = lagged_sum("levels_started", k)
            df = df.withColumn(f"rolling_completions_{k}d", comp) \
                   .withColumn(f"rolling_completion_rate_{k}d", F.when(starts > 0, comp / starts).otherwise(0.0))
    else:
        groups_skipped.append('rolling_completions')
        features_skipped.extend(rolling_completion_features)

    # 6) Skill stability (variance of avg attempts)
    skill_variance_features = [f"skill_variance_{k}d" for k in [3, 5]]
    if should_compute(*skill_variance_features):
        groups_computed.append('skill_variance')
        features_computed.extend(skill_variance_features)
        print(f"   ✅ Computing: skill_variance → {skill_variance_features}")
        for k in [3, 5]:
            df = df.withColumn(f"skill_variance_{k}d", lagged_std("current_avg_attemptperuser", k))
    else:
        groups_skipped.append('skill_variance')
        features_skipped.extend(skill_variance_features)

    # 7) Engagement streak: high engagement days in last 3/5 (global median threshold)
    high_engagement_features = [f"high_engagement_days_{k}d" for k in [3, 5]]
    if should_compute(*high_engagement_features):
        groups_computed.append('high_engagement')
        features_computed.extend(high_engagement_features)
        print(f"   ✅ Computing: high_engagement → {high_engagement_features}")
        # NOTE: Using fixed threshold instead of computing global median to avoid .collect() operations
        # Fixed threshold: 1800 seconds (30 minutes) - reasonable for "high engagement"
        median_play = 1800.0  # 30 minutes in seconds
        
        if median_play is not None:
            df = df.withColumn("__high_engagement__", F.when(F.col("daily_play_duration_seconds") >= F.lit(median_play), 1).otherwise(0))
            for k in [3, 5]:
                df = df.withColumn(f"high_engagement_days_{k}d", lagged_sum("__high_engagement__", k))
    else:
        groups_skipped.append('high_engagement')
        features_skipped.extend(high_engagement_features)

    # 8) Spending propensity indicators
    spending_features = ['is_spender_7d', 'days_since_last_spend', 'spending_frequency_7d', 
                        'avg_spend_per_active_day_7d', 'spending_volatility_7d']
    if should_compute(*spending_features):
        groups_computed.append('spending_propensity')
        features_computed.extend(spending_features)
        print(f"   ✅ Computing: spending_propensity → {spending_features}")
        spend_flag = F.when(F.col("daily_spend_coins") > 0, 1).otherwise(0)
        df = df.withColumn("__has_spend__", spend_flag)
        df = df.withColumn("is_spender_7d", F.max("__has_spend__").over(w.rowsBetween(-7, -1)))

        # Days since last spend (999 if never)
        spend_date = F.when(F.col("daily_spend_coins") > 0, F.col("session_date"))
        last_spend = F.max(spend_date).over(w.rowsBetween(Window.unboundedPreceding, -1))
        df = df.withColumn("days_since_last_spend", F.when(last_spend.isNull(), 999).otherwise(F.datediff(F.col("session_date"), last_spend)))

        # Spending frequency in last 7 days
        df = df.withColumn("spending_frequency_7d", lagged_sum("__has_spend__", 7) / F.lit(7.0))

        # Avg spend per active day in last 7 days
        spend_nonzero = F.when(F.col("daily_spend_coins") > 0, F.col("daily_spend_coins"))
        avg_spend_active_7 = F.avg(spend_nonzero).over(w.rowsBetween(-7, -1))
        df = df.withColumn("avg_spend_per_active_day_7d", F.coalesce(avg_spend_active_7, F.lit(0.0)))

        # Spending volatility in last 7 days
        df = df.withColumn("spending_volatility_7d", F.coalesce(lagged_std("daily_spend_coins", 7), F.lit(0.0)))
    else:
        groups_skipped.append('spending_propensity')
        features_skipped.extend(spending_features)

    # 9) Win/loss streaks and win/loss ratio (use completion_rate proxy)
    streak_features = ['consecutive_wins_streak', 'consecutive_losses_streak', 'win_loss_ratio_3d', 'recent_struggle_indicator']
    if should_compute(*streak_features):
        groups_computed.append('win_loss_streaks')
        features_computed.extend(streak_features)
        print(f"   ✅ Computing: win_loss_streaks → {streak_features}")
        win_flag = F.when(F.col("completion_rate") > 0.5, 1).otherwise(0)
        loss_flag = F.when(F.col("completion_rate") <= 0.5, 1).otherwise(0)
        df = df.withColumn("__win__", win_flag).withColumn("__loss__", loss_flag)

        # Consecutive wins streak (lagged)
        win_seg_start = F.when((F.col("__win__") == 1) & (F.lag("__win__", 1).over(w) != 1), 1).otherwise(0)
        win_seg_id = F.sum(win_seg_start).over(w.rowsBetween(Window.unboundedPreceding, 0))
        w_win_seg = Window.partitionBy("user_id", win_seg_id).orderBy("session_date")
        win_pos = F.when(F.col("__win__") == 1, F.row_number().over(w_win_seg)).otherwise(0)
        df = df.withColumn("__win_pos__", win_pos)
        df = df.withColumn("consecutive_wins_streak", F.coalesce(F.lag("__win_pos__", 1).over(w), F.lit(0)))

        # Consecutive losses streak (lagged)
        loss_seg_start = F.when((F.col("__loss__") == 1) & (F.lag("__loss__", 1).over(w) != 1), 1).otherwise(0)
        loss_seg_id = F.sum(loss_seg_start).over(w.rowsBetween(Window.unboundedPreceding, 0))
        w_loss_seg = Window.partitionBy("user_id", loss_seg_id).orderBy("session_date")
        loss_pos = F.when(F.col("__loss__") == 1, F.row_number().over(w_loss_seg)).otherwise(0)
        df = df.withColumn("__loss_pos__", loss_pos)
        df = df.withColumn("consecutive_losses_streak", F.coalesce(F.lag("__loss_pos__", 1).over(w), F.lit(0)))

        # Win/loss ratio (last 3 days, lagged)
        rolling_wins = lagged_sum("levels_completed", 3)
        rolling_attempts = lagged_sum("levels_started", 3)
        df = df.withColumn("win_loss_ratio_3d", F.when(rolling_attempts > 0, rolling_wins / rolling_attempts).otherwise(0.0))

        # Recent struggle indicator (yesterday completion_rate < 0.3)
        df = df.withColumn("recent_struggle_indicator", F.when(F.lag("completion_rate", 1).over(w) < 0.3, 1).otherwise(0))
    else:
        groups_skipped.append('win_loss_streaks')
        features_skipped.extend(streak_features)

    # 10) User lifecycle / cohort
    lifecycle_features = ['lifecycle_new_user', 'lifecycle_early', 'lifecycle_established', 'lifecycle_veteran', 'install_month']
    if should_compute(*lifecycle_features):
        groups_computed.append('lifecycle')
        features_computed.extend(lifecycle_features)
        print(f"   ✅ Computing: lifecycle → {lifecycle_features}")
        dsi = F.col("days_since_install")
        if "days_since_install" in df.columns:
            df = df.withColumn("lifecycle_new_user", F.when(dsi <= 7, 1).otherwise(0)) \
                   .withColumn("lifecycle_early", F.when((dsi > 7) & (dsi <= 30), 1).otherwise(0)) \
                   .withColumn("lifecycle_established", F.when((dsi > 30) & (dsi <= 90), 1).otherwise(0)) \
                   .withColumn("lifecycle_veteran", F.when(dsi > 90, 1).otherwise(0))

        # Install month from first session date per user (use partition-only window)
        w_user_all = Window.partitionBy("user_id")
        df = df.withColumn("install_month", F.month(F.min("session_date").over(w_user_all)))
    else:
        groups_skipped.append('lifecycle')
        features_skipped.extend(lifecycle_features)

    # Whale detection using lagged cumulative spend (cumsum shifted by 1)
    whale_features = ['total_lifetime_coins_spent', 'is_whale']
    if should_compute(*whale_features):
        groups_computed.append('whale_detection')
        features_computed.extend(whale_features)
        print(f"   ✅ Computing: whale_detection → {whale_features}")
        cum_spend_lagged = F.sum(F.col("daily_spend_coins")).over(w.rowsBetween(Window.unboundedPreceding, -1))
        df = df.withColumn("total_lifetime_coins_spent", F.coalesce(cum_spend_lagged, F.lit(0.0)))
        # NOTE: Using fixed threshold instead of computing global percentile to avoid .collect() operations
        whale_threshold = 10000.0  # Fixed threshold for whale classification
        
        if whale_threshold is not None:
            df = df.withColumn("is_whale", F.when(F.col("total_lifetime_coins_spent") >= F.lit(whale_threshold), 1).otherwise(0))
    else:
        groups_skipped.append('whale_detection')
        features_skipped.extend(whale_features)

    # 11) Difficulty trajectory
    difficulty_trajectory_features = ['difficulty_trend_3d', 'difficulty_change_magnitude_3d', 'time_at_current_difficulty', 'distance_from_comfort_zone']
    if should_compute(*difficulty_trajectory_features):
        groups_computed.append('difficulty_trajectory')
        features_computed.extend(difficulty_trajectory_features)
        print(f"   ✅ Computing: difficulty_trajectory → {difficulty_trajectory_features}")
        # Slope over last 3 days (lagged): use OLS slope formula on row index
        t = F.row_number().over(w)
        df = df.withColumn("__t__", t)
        y_mult = F.col("current_effectivelevelmultiplier")
        x = F.col("__t__")
        xy = x * y_mult
        x2 = x * x
        sum_x = F.sum(x).over(w.rowsBetween(-3, -1))
        sum_y_mult = F.sum(y_mult).over(w.rowsBetween(-3, -1))
        sum_xy = F.sum(xy).over(w.rowsBetween(-3, -1))
        sum_x2 = F.sum(x2).over(w.rowsBetween(-3, -1))
        n3 = F.count(F.lit(1)).over(w.rowsBetween(-3, -1))
        denom = (n3 * sum_x2 - sum_x * sum_x)
        slope_mult = F.when(denom > 0, (n3 * sum_xy - sum_x * sum_y_mult) / denom).otherwise(0.0)
        df = df.withColumn("difficulty_trend_3d", F.coalesce(slope_mult, F.lit(0.0)))

        # Difficulty change magnitude (sum abs diffs over last 3 days, lagged)
        mult_diff = F.abs(y_mult - F.lag(y_mult, 1).over(w))
        df = df.withColumn("__mult_diff__", mult_diff)
        df = df.withColumn("difficulty_change_magnitude_3d", F.coalesce(F.sum("__mult_diff__").over(w.rowsBetween(-3, -1)), F.lit(0.0)))

        # Time at current difficulty (consecutive same multiplier; lagged)
        prev_mult = F.lag(y_mult, 1).over(w)
        diff_seg_start = F.when(prev_mult.isNull() | (F.abs(y_mult - prev_mult) >= 0.01), 1).otherwise(0)
        diff_seg_id = F.sum(diff_seg_start).over(w.rowsBetween(Window.unboundedPreceding, 0))
        w_diff_seg = Window.partitionBy("user_id", diff_seg_id).orderBy("session_date")
        time_at_curr = F.row_number().over(w_diff_seg)
        df = df.withColumn("__time_at_diff__", time_at_curr)
        df = df.withColumn("time_at_current_difficulty", F.coalesce(F.lag("__time_at_diff__", 1).over(w), F.lit(0)))

        # Distance from comfort zone (current vs preferred over 7d)
        if "preferred_difficulty_7d" in df.columns:
            df = df.withColumn("distance_from_comfort_zone", F.abs(F.col("current_effectivelevelmultiplier") - F.col("preferred_difficulty_7d")))
    else:
        groups_skipped.append('difficulty_trajectory')
        features_skipped.extend(difficulty_trajectory_features)

    # 12) Engagement patterns
    engagement_features = ['session_intensity_3d', 'playtime_trend_3d', 'days_since_last_play', 'weekend_warrior']
    if should_compute(*engagement_features):
        groups_computed.append('engagement_patterns')
        features_computed.extend(engagement_features)
        print(f"   ✅ Computing: engagement_patterns → {engagement_features}")
        df = df.withColumn("session_intensity_3d", F.coalesce(lagged_avg("daily_event_count", 3), F.lit(0.0)))

        # Playtime trend (slope) over last 3 days (lagged) - need __t__ from difficulty trajectory
        if "__t__" not in df.columns:
            t = F.row_number().over(w)
            df = df.withColumn("__t__", t)
        x = F.col("__t__")
        y_play = F.col("daily_play_duration_seconds")
        xy_p = x * y_play
        sum_x = F.sum(x).over(w.rowsBetween(-3, -1))
        sum_y_p = F.sum(y_play).over(w.rowsBetween(-3, -1))
        sum_xy_p = F.sum(xy_p).over(w.rowsBetween(-3, -1))
        x2 = x * x
        sum_x2 = F.sum(x2).over(w.rowsBetween(-3, -1))
        n3 = F.count(F.lit(1)).over(w.rowsBetween(-3, -1))
        denom = (n3 * sum_x2 - sum_x * sum_x)
        slope_play = F.when(denom > 0, (n3 * sum_xy_p - sum_x * sum_y_p) / denom).otherwise(0.0)
        df = df.withColumn("playtime_trend_3d", F.coalesce(slope_play, F.lit(0.0)))

        # Days since last play (gap length, lagged)
        gap_curr = F.when(prev_date.isNull(), 0).otherwise(F.datediff(F.col("session_date"), prev_date) - 1)
        df = df.withColumn("__gap_curr__", gap_curr)
        df = df.withColumn("days_since_last_play", F.coalesce(F.lag("__gap_curr__", 1).over(w), F.lit(0)))

        # Weekend warrior (avg weekend flag last 7 days, lagged)
        df = df.withColumn("weekend_warrior", F.when(lagged_avg("is_weekend", 7) > 0.5, 1).otherwise(0))
    else:
        groups_skipped.append('engagement_patterns')
        features_skipped.extend(engagement_features)

    # 13) Cross-feature interactions (current day context)
    interaction_features = ['skill_x_difficulty', 'spend_x_completion', 'playtime_x_skill']
    if should_compute(*interaction_features):
        groups_computed.append('interactions')
        features_computed.extend(interaction_features)
        print(f"   ✅ Computing: interactions → {interaction_features}")
        df = df.withColumn("skill_x_difficulty", F.col("current_avg_attemptperuser") * F.col("current_effectivelevelmultiplier")) \
               .withColumn("spend_x_completion", F.col("daily_spend_coins") * F.col("completion_rate")) \
               .withColumn("playtime_x_skill", F.col("daily_play_duration_seconds") * F.col("current_avg_attemptperuser"))
    else:
        groups_skipped.append('interactions')
        features_skipped.extend(interaction_features)

    # Cleanup temp columns
    for c in [
        "__seg_id_days__", "__streak_days_curr__", "__high_engagement__",
        "__win__", "__loss__", "__win_pos__", "__loss_pos__",
        "__t__", "__mult_diff__", "__time_at_diff__", "__gap_curr__", "__has_spend__"
    ]:
        if c in df.columns:
            df = df.drop(c)

    # Fill NaNs in newly added advanced features to 0 (parity with pandas fillna(0))
    new_cols = [c for c in df.columns if c not in original_cols]
    numeric_types = {"int", "bigint", "double", "float", "decimal", "smallint", "tinyint"}
    new_numeric_cols = [c for c, t in df.dtypes if c in new_cols and any(nt in t for nt in numeric_types)]
    if new_numeric_cols:
        df = df.fillna(0, subset=new_numeric_cols)

    # Print detailed summary
    print("-"*80)
    if feature_whitelist:
        print(f"✅ ADVANCED FEATURES COMPUTATION COMPLETE (OPTIMIZED MODE)")
        print(f"   Groups computed: {len(groups_computed)} → {groups_computed}")
        print(f"   Groups skipped:  {len(groups_skipped)} → {groups_skipped}")
        print(f"   Features computed: {len(features_computed)}")
        print(f"   Features skipped:  {len(features_skipped)}")
        print(f"   New columns added to DataFrame: {len(new_cols)}")
    else:
        print(f"✅ ADVANCED FEATURES COMPUTATION COMPLETE (FULL MODE)")
        print(f"   All {len(groups_computed)} feature groups computed")
        print(f"   New columns added: {len(new_cols)}")
    print("="*80 + "\n")
    return df


def write_partitioned_dataset(spark: SparkSession,
                              df: DataFrame,
                              out_dir: str,
                              partition_col: str,
                              fmt: str = "delta",
                              target_partitions: Optional[int] = None,
                              max_records_per_file: int = 0,
                              write_mode: str = "overwrite",
                              replace_partition_date: Optional[str] = None,
                              replace_partition_end_date: Optional[str] = None) -> None:
    """Write dataset in Delta or Parquet, partitioned by a single column (e.g., session_date)."""
    fmt = fmt.lower()
    if fmt not in {"delta", "parquet"}:
        raise ValueError("--output-format must be 'delta' or 'parquet'")

    print(f"💾 Writing {fmt.upper()} to {out_dir} partitioned by [{partition_col}]...")
    print(f"   Write mode: {write_mode}")
    
    # Optimize repartitioning based on available memory
    # For larger clusters, we can use more partitions for better parallelism
    if target_partitions is None or target_partitions <= 0:
        target_partitions = max(200, int(spark.conf.get("spark.sql.shuffle.partitions", "200")))

    print(f"   Using {target_partitions} output partitions (before partitionBy '{partition_col}')")
    df = df.repartition(target_partitions, F.col(partition_col))
    
    # Determine write mode
    if write_mode == "replaceWhere":
        if fmt != "delta":
            raise ValueError("replaceWhere mode only supported for Delta format")
        
        if replace_partition_date and replace_partition_end_date:
             predicate = f"{partition_col} >= '{replace_partition_date}' AND {partition_col} <= '{replace_partition_end_date}'"
        elif replace_partition_date:
             predicate = f"{partition_col} = '{replace_partition_date}'"
        else:
             raise ValueError("replaceWhere requires replace_partition_date (and optional replace_partition_end_date)")

        writer_base = (
            df.write
              .format("delta")
              .mode("overwrite")
              .option("replaceWhere", predicate)
              .option("compression", "snappy")
              .option("mergeSchema", "true")
              .partitionBy(partition_col)
        )
    else:
        # Standard overwrite or append
        writer_base = (
            df.write
              .mode(write_mode)
              .option("compression", "snappy")
              .option("mergeSchema", "true")
              .option("overwriteSchema", "true" if write_mode == "overwrite" else "false")
              .partitionBy(partition_col)
        )
    if max_records_per_file and max_records_per_file > 0:
        writer_base = writer_base.option("maxRecordsPerFile", str(max_records_per_file))

    spark_out_dir = to_spark_path(out_dir)

    if fmt == "delta":
        try:
            # Disable OPTIMIZE during write to reduce memory pressure
            # OPTIMIZE can be run separately after write completes
            writer_base.format("delta").save(spark_out_dir)
            print("✅ Delta write complete")
            print("   Note: OPTIMIZE skipped during write to reduce memory pressure")
            print(f"   Consider running OPTIMIZE separately: OPTIMIZE delta.`{out_dir}` ZORDER BY (user_id)")
        except Exception as e:
            msg = str(e)
            error_str = str(msg).lower()
            # Check if it's OOM or Delta exception handler error
            is_oom = "DriverStoppedException" in msg or "exit code: 134" in msg or "OOM" in error_str or "out of memory" in error_str
            is_delta_error = "delta.exceptions.captured" in msg or "module named 'delta.exceptions" in msg
            
            if is_oom:
                print(f"⚠️  Delta write failed due to OOM: {msg}")
                print("   Delta writes are too memory-intensive for this cluster.")
                print("   Will fall back to Parquet format instead (less memory-intensive).")
            elif is_delta_error:
                print(f"⚠️  Delta write failed due to exception handler issue: {msg}")
                print("   This is likely a Delta Lake library compatibility issue.")
                print("   Delta writes are not working due to library incompatibility.")
                print("   Will fall back to Parquet format instead.")
            
            if not is_oom and not is_delta_error:
                print(f"⚠️  Delta write failed: {msg}")
                print("   Falling back to Parquet write...")
            
            # For Parquet fallback, use a different path to avoid Delta table conflicts
            # Add .parquet suffix to the output directory
            parquet_out_dir = out_dir.replace('.delta', '.parquet') if out_dir.endswith('.delta') else f"{out_dir}.parquet"
            spark_parquet_dir = to_spark_path(parquet_out_dir)
            
            # Coalesce even more aggressively for Parquet fallback
            df = df.repartition(target_partitions, F.col(partition_col))
            
            # Thoroughly clean up the Delta table directory including _delta_log
            try:
                dbutils  # Check if in Databricks
                driver_out_dir = to_driver_dbfs_path(out_dir)
                dbfs_path = driver_out_dir.replace('/dbfs/', '/') if driver_out_dir.startswith('/dbfs/') else driver_out_dir
                # Remove the entire directory recursively
                dbutils.fs.rm(dbfs_path, True)
                # Also explicitly remove _delta_log if it exists
                delta_log_path = f"{dbfs_path}/_delta_log"
                try:
                    dbutils.fs.rm(delta_log_path, True)
                except:
                    pass  # _delta_log might already be removed
                print(f"   Cleaned up existing Delta table at {out_dir}")
            except NameError:
                # Not in Databricks, use os
                import os
                import shutil
                driver_out_dir = to_driver_dbfs_path(out_dir)
                if os.path.exists(driver_out_dir):
                    shutil.rmtree(driver_out_dir)
                    # Also remove _delta_log explicitly
                    delta_log_path = os.path.join(driver_out_dir, "_delta_log")
                    if os.path.exists(delta_log_path):
                        shutil.rmtree(delta_log_path)
                    print(f"   Cleaned up existing directory at {out_dir}")
            except Exception as cleanup_ex:
                print(f"   Warning: Could not clean up existing directory: {cleanup_ex}")
            
            # Use Parquet format with different output path to avoid conflicts
            print(f"   Writing Parquet to: {parquet_out_dir}")
            try:
                writer_base.format("parquet").save(spark_parquet_dir)
                print("✅ Parquet write complete (fallback)")
                print(f"⚠️  Note: Output is at {parquet_out_dir} instead of {out_dir}")
            except Exception as parquet_error:
                error_msg = str(parquet_error)
                # Check if it's the Delta exception handler error
                if "delta.exceptions.captured" in error_msg or "ModuleNotFoundError" in error_msg:
                    print(f"⚠️  Parquet write hit Delta exception handler issue: {error_msg}")
                    print("   Attempting direct Parquet write without format()...")
                    try:
                        # Try direct parquet() method with coalesce to reduce memory pressure
                        # Coalesce to fewer partitions to reduce memory during write
                        # Avoid checking partition count (triggers materialization) - just coalesce directly
                        print(f"   Repartitioning to 100 partitions for Parquet write...")
                        df = df.repartition(100, F.col(partition_col))
                        
                        # Use direct parquet() method - this bypasses format() entirely
                        df.write \
                          .mode("overwrite") \
                          .option("compression", "snappy") \
                          .partitionBy(partition_col) \
                          .parquet(spark_parquet_dir)
                        print("✅ Parquet write complete (direct method)")
                        print(f"⚠️  Note: Output is at {parquet_out_dir} instead of {out_dir}")
                    except Exception as direct_error:
                        direct_error_msg = str(direct_error)
                        # Check if it's still the Delta exception handler
                        if "delta.exceptions.captured" in direct_error_msg:
                            print(f"❌ Delta exception handler still interfering: {direct_error_msg}")
                            print("   This is a Delta Lake library compatibility issue.")
                            print("   Solution: Install Delta Python package:")
                            print("   - In cluster libraries: Install 'delta-spark==3.0.0' from PyPI")
                            print("   - Or run: %pip install delta-spark==3.0.0")
                            print("   Then restart cluster and try again.")
                        else:
                            print(f"❌ Direct Parquet write failed: {direct_error_msg}")
                            print("   This may be a different issue (OOM, permissions, etc.)")
                        raise
                else:
                    # Different error, re-raise
                    raise
    else:
        writer_base.parquet(spark_out_dir)
        print("✅ Parquet write complete")


def lowercase_all_columns(df: DataFrame) -> DataFrame:
    """Alias all columns to lowercase for consistent schema."""
    return df.select(*[F.col(c).alias(c.lower()) for c in df.columns])


def _spark_dtype_for(dt: DataType) -> DataType:
    """Utility to coerce nested numeric types to DoubleType for EWMA outputs."""
    if isinstance(dt, (IntegerType, LongType, FloatType, DecimalType)):
        return DoubleType()
    return dt


def compute_ewma_features_spark(df: DataFrame,
                                alpha_values: List[float] = [0.1, 0.3, 0.5, 0.7],
                                ewma_cols: List[str] = None,
                                ewma_filter: Dict[str, Set[float]] = None) -> DataFrame:
    """
    Compute lagged EWMA features per user using a grouped map Pandas UDF (mapInPandas).
    - Strictly runs inside Spark executors via Arrow; no driver toPandas.
    - Parity with pandas: ewma(...).shift(1)
    - Excludes identifiers, lag features, and any 'next_*' columns.
    - If ewma_cols is provided, only those columns are used (reduces memory pressure).
    - If ewma_filter is provided (inference mode), only compute specific (column, alpha) pairs.
    Returns a DataFrame with ['user_id','session_date', ewma_* columns].
    """
    # INFERENCE MODE OPTIMIZATION: If ewma_filter is provided, only compute needed EWMA features
    if ewma_filter:
        # Extract the specific columns and alphas needed
        filtered_cols = list(ewma_filter.keys())
        # Get union of all alphas needed
        all_alphas_needed = set()
        for alphas in ewma_filter.values():
            all_alphas_needed.update(alphas)
        alpha_values = sorted(list(all_alphas_needed))
        
        # CASE-INSENSITIVE column matching: DataFrame columns may be mixed case
        # but selected_features.json uses lowercase
        available_cols = set(df.columns)
        available_cols_lower = {c.lower(): c for c in df.columns}  # lowercase -> actual name
        
        # Map filtered columns to actual DataFrame column names (case-insensitive)
        numeric_cols = []
        col_name_mapping = {}  # lowercase -> actual column name in DataFrame
        for col in filtered_cols:
            col_lower = col.lower()
            if col_lower in available_cols_lower:
                actual_col = available_cols_lower[col_lower]
                numeric_cols.append(actual_col)
                col_name_mapping[col] = actual_col
        
        # Also update ewma_filter keys to use actual column names
        if col_name_mapping:
            updated_ewma_filter = {}
            for col, alphas in ewma_filter.items():
                if col in col_name_mapping:
                    updated_ewma_filter[col_name_mapping[col]] = alphas
            ewma_filter = updated_ewma_filter
        
        if not numeric_cols:
            print(f"⚠️  EWMA filter specified {len(filtered_cols)} columns but none found in dataframe")
            print(f"   Filter columns: {filtered_cols[:5]}...")
            print(f"   Available columns sample: {list(available_cols)[:5]}...")
            print(f"   Falling back to auto-detect mode")
            ewma_filter = None  # Fall back to normal mode
        else:
            total_features = sum(len(ewma_filter.get(col, set())) for col in numeric_cols)
            print(f"📊 Computing EWMA features (OPTIMIZED INFERENCE MODE)...")
            print(f"   Only computing {total_features} EWMA features (vs 680+ in full mode)")
            print(f"   Columns: {len(numeric_cols)}, Alphas: {alpha_values}")
            if len(numeric_cols) < len(filtered_cols):
                missing = set(filtered_cols) - set(col_name_mapping.keys())
                print(f"   ⚠️  {len(missing)} columns not found: {list(missing)[:5]}...")
    
    if not ewma_filter:
        print(f"📊 Computing EWMA features via grouped Pandas UDF for alphas={alpha_values}...")

    # Ensure we have normalized keys present
    assert 'user_id' in df.columns and 'session_date' in df.columns, "Expected user_id and session_date columns"

    # Determine numeric columns to include in EWMA (if not already set by filter)
    exclude = {"user_id", "session_date", "difficulty_arm", "previous_day_action", "previous_day_multiplier"}
    numeric_types = (IntegerType, LongType, FloatType, DoubleType, DecimalType)
    
    if ewma_filter:
        # Already set numeric_cols above
        pass
    elif ewma_cols:
        # Use curated column list
        available_cols = set(df.columns)
        numeric_cols = [col for col in ewma_cols if col in available_cols and not col.startswith('next_') and col not in exclude]
        print(f"   Using curated EWMA columns ({len(numeric_cols)}): {', '.join(numeric_cols[:10])}{'...' if len(numeric_cols) > 10 else ''}")
    else:
        # Auto-detect all numeric columns (original behavior)
        numeric_cols = [name for name, dtype in df.dtypes
                        if not name.startswith('next_') and name not in exclude and any(t in dtype for t in ['int', 'bigint', 'double', 'float', 'decimal', 'smallint', 'tinyint'])]
        print(f"   Auto-detected {len(numeric_cols)} numeric columns for EWMA")

    if not numeric_cols:
        print("   No numeric columns found for EWMA; skipping.")
        return df

    # Build schema for pandas UDF output
    # Keep user_id/session_date types from input; EWMA cols as DoubleType
    user_id_field = next((f for f in df.schema.fields if f.name == 'user_id'), None)
    session_date_field = next((f for f in df.schema.fields if f.name == 'session_date'), None)
    user_id_type = user_id_field.dataType if user_id_field else StringType()
    session_date_type = session_date_field.dataType if session_date_field else DateType()

    # Build EWMA fields based on filter or full computation
    ewma_fields = []
    col_alpha_pairs = []  # Track which (col, alpha) pairs to compute
    
    for col in numeric_cols:
        if ewma_filter:
            # Only include alphas needed for this column
            alphas_for_col = ewma_filter.get(col, set())
            for a in sorted(alphas_for_col):
                ewma_fields.append(StructField(f"ewma_{col}_alpha{int(a*10)}", DoubleType(), True))
                col_alpha_pairs.append((col, a))
        else:
            # Include all alphas for this column
            for a in alpha_values:
                ewma_fields.append(StructField(f"ewma_{col}_alpha{int(a*10)}", DoubleType(), True))
                col_alpha_pairs.append((col, a))

    ewma_schema = StructType([
        StructField('user_id', user_id_type, False),
        StructField('session_date', session_date_type, False),
        *ewma_fields
    ])

    # Create closure variables for the UDF
    _col_alpha_pairs = col_alpha_pairs
    _numeric_cols = numeric_cols
    
    def ewma_apply(pdf):
        import pandas as pd
        from datetime import date as date_type
        if 'session_date' in pdf.columns:
            pdf = pdf.sort_values(['session_date'])
        # ROBUST session_date conversion to ensure join compatibility
        # Always convert to Python date objects to match Spark DateType
        raw_dates = pdf['session_date']
        dtype_str = str(raw_dates.dtype)
        
        # Convert based on the actual type
        if pd.api.types.is_datetime64_any_dtype(raw_dates) or dtype_str.startswith('datetime'):
            # datetime64 or datetime-like: extract date part
            session_dates = pd.to_datetime(raw_dates).dt.date.values
        elif raw_dates.dtype == 'object':
            # Object dtype: could be date objects, datetime objects, or strings
            # Try to convert each element to date
            def to_date(x):
                if isinstance(x, date_type):
                    return x
                elif hasattr(x, 'date'):  # datetime object
                    return x.date()
                else:  # string or other
                    try:
                        return pd.to_datetime(x).date()
                    except:
                        return x
            session_dates = raw_dates.apply(to_date).values
        else:
            # Fallback: try pandas conversion
            try:
                session_dates = pd.to_datetime(raw_dates).dt.date.values
            except:
                session_dates = raw_dates.values
        
        out = pd.DataFrame({
            'user_id': pdf['user_id'].values,
            'session_date': session_dates
        })
        # Compute only the needed (col, alpha) pairs
        computed_series = {}  # Cache series to avoid recomputing
        for col, a in _col_alpha_pairs:
            if col not in computed_series:
                computed_series[col] = pd.to_numeric(pdf[col], errors='coerce')
            series = computed_series[col]
            out[f"ewma_{col}_alpha{int(a*10)}"] = (
                series.ewm(alpha=a, adjust=False).mean().shift(1)
            )
        for c in out.columns:
            if c not in ('user_id', 'session_date'):
                out[c] = out[c].fillna(0.0)
        return out

    # Group by user and compute EWMAs
    ewma_df = df.groupBy('user_id').applyInPandas(ewma_apply, schema=ewma_schema)

    # DEBUG: Check session_date types and sample values
    orig_session_date_type = df.schema['session_date'].dataType
    ewma_session_date_type = ewma_df.schema['session_date'].dataType
    print(f"   🔍 DEBUG: Original session_date type: {orig_session_date_type}")
    print(f"   🔍 DEBUG: EWMA session_date type: {ewma_session_date_type}")
    
    # Join back with original df on keys
    # Use LEFT join to preserve all original rows (INNER join can drop rows due to type mismatches)
    joined = df.join(ewma_df, on=['user_id', 'session_date'], how='left')
    
    # Fill any missing EWMA values with 0 (can happen if join fails for some rows)
    ewma_col_names = [f.name for f in ewma_schema.fields if f.name not in ('user_id', 'session_date')]
    for col_name in ewma_col_names:
        if col_name in joined.columns:
            joined = joined.withColumn(col_name, F.coalesce(F.col(col_name), F.lit(0.0)))
    
    print("✅ EWMA features computed and joined")
    return joined


def normalize_path(path: str) -> str:
    """
    Convert relative paths to absolute DBFS paths for Databricks.
    In Databricks, all Spark paths must be absolute.
    """
    if not path:
        return path
    # If already absolute (starts with / or dbfs:), return as-is
    if path.startswith('/') or path.startswith('dbfs:'):
        return path
    # Check if we're in Databricks
    try:
        dbutils  # If dbutils exists, we're in Databricks
        # Convert relative path to DBFS absolute path
        if not path.startswith('/dbfs/'):
            return f'/dbfs/{path}'
        return path
    except NameError:
        # Not in Databricks, convert to absolute path using os.path
        import os
        return os.path.abspath(path)


def to_spark_path(path: str) -> str:
    """Convert driver-style /dbfs paths to spark-friendly dbfs:/ URIs."""
    if not path:
        return path
    if path.startswith("/dbfs/"):
        return "dbfs:" + path[5:]
    return path


def to_driver_dbfs_path(path: str) -> str:
    """Convert dbfs:/ URIs to /dbfs paths accessible from the driver."""
    if not path:
        return path
    if path.startswith("dbfs:/"):
        return "/dbfs" + path[5:]
    return path


def delete_dbfs_path(path: str) -> None:
    """Delete a DBFS path (dbfs:/ or /dbfs/) recursively if it exists."""
    normalized = normalize_path(path)
    driver_path = to_driver_dbfs_path(normalized) or normalized
    if not driver_path:
        return
    print(f"🧹 Removing existing path before re-run: {driver_path}")
    try:
        dbutils  # type: ignore
        dbfs_path = driver_path.replace('/dbfs/', '/') if driver_path.startswith('/dbfs/') else driver_path
        dbutils.fs.rm(dbfs_path, True)  # type: ignore
    except NameError:
        import os
        import shutil
        if os.path.exists(driver_path):
            shutil.rmtree(driver_path)
    except Exception as exc:
        print(f"   ⚠️  Failed to remove {driver_path}: {exc}")


def get_executor_memory_bytes(spark) -> int:
    """
    Best-effort detection of per-executor memory in bytes.

    Prefers SparkContext executor status; falls back to spark.executor.memory or 8GB default.
    """
    try:
        statuses = spark.sparkContext.getExecutorMemoryStatus()
        mem_values = [mem[0] for mem in statuses.values() if mem[0] > 0]
        if mem_values:
            return min(mem_values)
    except Exception:
        pass

    # Fallback to spark.executor.memory config
    try:
        mem_conf = spark.conf.get("spark.executor.memory", "0")
        mem_conf = mem_conf.strip().lower()
        if mem_conf.endswith('g'):
            return int(float(mem_conf[:-1])) * 1024 ** 3
        if mem_conf.endswith('m'):
            return int(float(mem_conf[:-1])) * 1024 ** 2
        if mem_conf.isdigit():
            return int(mem_conf)
    except Exception:
        pass

    return 8 * 1024 ** 3  # default 8GB


def estimate_avg_row_size(df, sample_rows: int = 200000) -> float:
    """Estimate average row size (bytes) using a JSON sample."""
    try:
        sample_df = df.limit(sample_rows)
        json_rdd = sample_df.toJSON().rdd.map(lambda s: len(s.encode('utf-8')))
        count = json_rdd.count()
        if count == 0:
            return 0.0
        total = json_rdd.sum()
        return float(total) / count
    except Exception:
        return 0.0


def plan_partitions_for_direct_write(
    df,
    spark,
    base_partitions: int,
    max_partitions_by_cluster: int,
    sample_rows: int = 200000,
    safety_fraction: float = 0.25,
) -> tuple[int, dict]:
    """
    Determine a safe number of partitions for direct Delta write by estimating row size.

    Returns (partitions, metrics_dict)
    """
    avg_row_size = estimate_avg_row_size(df, sample_rows=sample_rows)
    metrics = {
        "avg_row_size": avg_row_size,
        "row_count": 0,
        "safe_bytes_per_partition": 0,
        "estimated_bytes": 0,
        "executor_memory": 0,
    }

    row_count = df.count()
    metrics["row_count"] = row_count

    if avg_row_size <= 0:
        avg_row_size = 1024.0  # fallback 1KB per row

    estimated_bytes = avg_row_size * row_count
    metrics["estimated_bytes"] = estimated_bytes

    executor_mem_bytes = get_executor_memory_bytes(spark)
    metrics["executor_memory"] = executor_mem_bytes
    safe_bytes = max(64 * 1024 ** 2, int(executor_mem_bytes * safety_fraction))
    metrics["safe_bytes_per_partition"] = safe_bytes

    desired_partitions = base_partitions
    if estimated_bytes > 0 and safe_bytes > 0:
        desired_partitions = max(
            base_partitions,
            int(math.ceil(estimated_bytes / safe_bytes))
        )

    final_partitions = min(max(desired_partitions, base_partitions), max_partitions_by_cluster)
    return final_partitions, metrics


def write_checkpoint_dataset(spark: SparkSession,
                            df: DataFrame,
                            checkpoint_dir: str,
                            name: str,
                            partition_col: str,
                            fmt: str) -> str:
    # Normalize to absolute path
    checkpoint_dir = normalize_path(checkpoint_dir)
    driver_checkpoint_dir = to_driver_dbfs_path(checkpoint_dir)
    
    # Create directory - use dbutils in Databricks, os in local
    try:
        dbutils  # Check if in Databricks
        # Use dbfs path without /dbfs prefix for dbutils
        driver_path = driver_checkpoint_dir if driver_checkpoint_dir else checkpoint_dir
        dbfs_path = driver_path.replace('/dbfs/', '/') if driver_path.startswith('/dbfs/') else driver_path
        dbutils.fs.mkdirs(dbfs_path)
    except NameError:
        # Local execution
        import os
        os.makedirs(driver_checkpoint_dir or checkpoint_dir, exist_ok=True)
    except Exception:
        pass  # Directory might already exist
    
    # Ensure os is available for path.join
    import os
    base_dir = driver_checkpoint_dir or checkpoint_dir
    out_dir = os.path.join(base_dir, f"{name}.{fmt}")
    print(f"💾 Writing checkpoint [{name}] to {out_dir} ({fmt})...")
    try:
        shuffle_default = int(spark.conf.get("spark.sql.shuffle.partitions", "200"))
    except Exception:
        shuffle_default = 200
    checkpoint_partitions = max(100, shuffle_default // 2)
    write_partitioned_dataset(
        spark,
        df,
        out_dir,
        partition_col=partition_col,
        fmt=fmt,
        target_partitions=checkpoint_partitions,
    )
    return out_dir


def read_checkpoint_dataset(spark: SparkSession,
                           checkpoint_dir: str,
                           name: str,
                           fmt: str):
    """Read a checkpoint with robust fallback across Delta/Parquet suffixes."""
    # Normalize to absolute path
    checkpoint_dir = normalize_path(checkpoint_dir)
    driver_checkpoint_dir = to_driver_dbfs_path(checkpoint_dir) or checkpoint_dir
    # Ensure os is available
    import os
    base = os.path.join(driver_checkpoint_dir, name)
    # primary path with requested fmt
    primary_path = f"{base}.{fmt}"
    spark_primary_path = to_spark_path(primary_path)
    print(f"📥 Reading checkpoint [{name}] from {primary_path} ({fmt})...")
    try:
        reader_fmt = 'parquet' if fmt == 'parquet' else 'delta'
        return spark.read.format(reader_fmt).load(spark_primary_path)
    except Exception as e:
        print(f"⚠️  Primary read failed: {e}")
        # try alternate suffix
        alt_fmt = 'parquet' if fmt == 'delta' else 'delta'
        alt_path = f"{base}.{alt_fmt}"
        spark_alt_path = to_spark_path(alt_path)
        print(f"🔁 Trying alternate checkpoint path: {alt_path} ({alt_fmt})")
        try:
            reader_fmt = 'parquet' if alt_fmt == 'parquet' else 'delta'
            return spark.read.format(reader_fmt).load(spark_alt_path)
        except Exception as e2:
            # final attempt: read parquet at primary folder regardless of suffix
            try:
                print(f"🔁 Final attempt: reading as Parquet at {primary_path}")
                return spark.read.parquet(spark_primary_path)
            except Exception:
                print(f"❌ Failed to read checkpoint [{name}] from {primary_path} or {alt_path}")
                raise e2


def main():
    # Fix: Ensure delta-spark Python package is loaded (not JAR stub)
    # This prevents ModuleNotFoundError for delta.exceptions.captured
    # NOTE: Skip if we're in a Databricks job (may cause OOM if done too early)
    try:
        import delta
        # Check if loading from JAR (wrong) instead of Python package (correct)
        if '.jar' in delta.__file__:
            print("⚠️  Delta loaded from JAR, installing Python package to take precedence...")
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", "--no-deps", "delta-spark==3.0.0"], 
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # Force reload delta module
            import importlib
            if 'delta' in sys.modules:
                del sys.modules['delta']
            import delta
            print(f"✅ Delta Python package installed and loaded from: {delta.__file__}")
        else:
            # Verify the exceptions module exists (but skip if in JAR path)
            import os
            delta_file = delta.__file__
            if '.jar' not in delta_file:
                delta_dir = os.path.dirname(delta_file)
                captured_file = os.path.join(delta_dir, 'exceptions', 'captured.py')
                if not os.path.exists(captured_file):
                    print("⚠️  delta.exceptions.captured module missing, attempting to fix...")
                    import subprocess
                    import sys
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "--force-reinstall", "delta-spark==3.0.0"],
                                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    # Reload
                    import importlib
                    if 'delta' in sys.modules:
                        del sys.modules['delta']
                    import delta
    except Exception as e:
        print(f"⚠️  Could not verify/fix delta package: {e}")
        print("   Continuing anyway - may hit Delta exception handler errors")
    
    parser = argparse.ArgumentParser(description="Phase 0: Data Generation from Snowflake for VW Bandit Pipeline (PURE SPARK)")
    parser.add_argument('--use-parquet', action='store_true', help='Load from Parquet instead of Snowflake')
    parser.add_argument('--use-csv', action='store_true', help='Load from CSV instead of Snowflake')
    parser.add_argument('--parquet-path', type=str, default='./data/spaceplay_raw.parquet', help='Path to Parquet export')
    parser.add_argument('--csv-path', type=str, default='./data/spaceplay_raw.csv', help='Path to CSV export directory')
    parser.add_argument('--out-dir', type=str, default='dbfs:/mnt/bandit/data/daily_features_spark.delta', help='Output dataset directory (Delta by default). Use dbfs:/ or /dbfs/ for DBFS paths.')
    parser.add_argument('--start-date', type=str, default='2025-07-01', help='Minimum gauserstartdate for Snowflake query (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None, help='Maximum gauserstartdate for Snowflake query (YYYY-MM-DD). If not set, defaults to now.')
    parser.add_argument('--output-format', type=str, default='delta', choices=['delta','parquet'], help='Output format for dataset')
    parser.add_argument('--partition-column', type=str, default='session_date', help='Column to partition the output by')
    parser.add_argument('--write-mode', type=str, default='overwrite', choices=['overwrite', 'append', 'replaceWhere'], help='Write mode: overwrite (entire table), append (add partitions), replaceWhere (replace specific partition)')
    parser.add_argument('--replace-partition-date', type=str, help='When --write-mode=replaceWhere, replace partition for this date (YYYY-MM-DD)')
    parser.add_argument('--shuffle-partitions', type=int, default=2000, help='spark.sql.shuffle.partitions value')
    parser.add_argument('--write-partitions', type=int, default=400, help='Number of partitions to use when writing output (repartitioned on partition column)')
    parser.add_argument('--local-dirs', type=str, default='', help='Comma-separated local dirs for shuffle spill (spark.local.dir). WARNING: /tmp may be small/in-memory; use large disk partitions for big runs.')
    parser.add_argument('--disable-aqe', action='store_true', help='Disable Adaptive Query Execution (AQE)')
    parser.add_argument('--force-local-spark', action='store_true', help='Force Spark master=local[*] (use only for laptop debugging; leave off on Databricks)')
    parser.add_argument('--max-records-per-file', type=int, default=0, help='Optional cap for records per output file (helps avoid large file sizes/OOM). 0 = disabled')
    # EWMA options for scale
    parser.add_argument('--disable-ewma', action='store_true', help='Skip EWMA feature computation (saves memory/time on large runs)')
    parser.add_argument('--ewma-cols', type=str, default='', help='Comma-separated list of columns to compute EWMA on (if not provided, all numeric columns are used). Ignored if --disable-ewma is set.')
    parser.add_argument('--ewma-lookback-days', type=int, default=30, help='Number of days to look back for EWMA computation in inference mode. EWMA with alpha=0.1 needs ~30 days for 95%% weight coverage.')
    # Checkpointing options
    parser.add_argument('--enable-checkpoints', action='store_true', help='Enable writing intermediate checkpoints (daily_features, daily_bandit, advanced_features)')
    parser.add_argument('--checkpoint-dir', type=str, default='dbfs:/mnt/bandit/checkpoints', help='Directory to store checkpoints. Use dbfs:/ or /dbfs/ for DBFS paths.')
    parser.add_argument('--resume-from', type=str, default='none', choices=['none','daily_features','daily_bandit','advanced_features'], help='Resume from an intermediate checkpoint')
    parser.add_argument('--skip-advanced-materialization', action='store_true', help='Skip advanced feature materialization (useful when /dbfs temp writes crash the driver)')
    parser.add_argument('--skip-final-materialization', action='store_true', help='Skip final dataset materialization before write (prevents OOM on large datasets)')
    parser.add_argument('--skip-checkpoint', action='store_true', help='Skip checkpoint before final write (write directly to Delta without breaking lineage)')
    parser.add_argument('--final-write-partitions', type=int, default=0, help='Override number of partitions for final Delta write when skipping checkpoint (legacy behavior; defaults to partitions-per-day heuristic when 0).')
    parser.add_argument('--partitions-per-day', type=int, default=2000, help='Multiplier used to determine final partitions (partitions_per_day × distinct session dates). Ignored if --final-write-partitions > 0.')
    parser.add_argument('--partition-size-sample', type=int, default=200000, help='Rows to sample when estimating average row size for partition planning (default: 200k)')
    parser.add_argument('--materialize-storage-level', type=str, default='DISK_ONLY', choices=['MEMORY_ONLY','MEMORY_AND_DISK','DISK_ONLY','MEMORY_ONLY_SER','MEMORY_AND_DISK_SER','DISK_ONLY_2','NONE'], help='StorageLevel to use when caching advanced features prior to write (default DISK_ONLY).')
    parser.add_argument('--materialize-partitions', type=int, default=200, help='Number of partitions to use when materializing advanced features (ignored if skipping).')
    # Snowflake connector options
    parser.add_argument('--use-sf-connector', action='store_true', help='Use Spark Snowflake Connector instead of JDBC')
    parser.add_argument('--sf-url', type=str, default='', help='Snowflake URL, e.g., https://<account>.snowflakecomputing.com (overrides JSON)')
    parser.add_argument('--sf-user', type=str, default='', help='Snowflake user (overrides JSON)')
    parser.add_argument('--sf-password', type=str, default='', help='Snowflake password (overrides JSON)')
    parser.add_argument('--sf-account', type=str, default='', help='Snowflake account (optional if derivable; overrides JSON)')
    parser.add_argument('--sf-warehouse', type=str, default='ML_DATABRICKS', help='Snowflake warehouse')
    parser.add_argument('--sf-database', type=str, default='SPACEPLAY', help='Snowflake database')
    parser.add_argument('--sf-schema', type=str, default='UNITY', help='Snowflake schema')
    parser.add_argument('--sf-role', type=str, default='', help='Snowflake role (optional)')
    parser.add_argument('--sf-packages', type=str, default='net.snowflake:spark-snowflake_2.13:3.1.4,net.snowflake:snowflake-jdbc:3.24.2', help='Maven coordinates for Snowflake connector packages (match Spark/Scala). Updated to 3.24.2 for compatibility')
    parser.add_argument('--delta-packages', type=str, default='io.delta:delta-spark_2.13:4.0.0', help='Maven coordinates for Delta Lake (match Spark/Scala)')
    parser.add_argument('--sf-config-json', type=str, default='config/snowflake.json', help='Path to Snowflake JSON config (auto-loaded if present)')
    parser.add_argument('--snowflake-table', type=str, default=None, help='Snowflake table name (e.g., spaceplay.unity.boxjam_daily_2025_10_20). If not provided, uses default snapshot table.')
    parser.add_argument('--inference', action='store_true', help='Inference mode: triggers Phase 6→5 job chain after completion. Use with --write-mode replaceWhere.')
    
    # Training mode triggers (Phase 0 → 1 → 2 → 3 → 4)
    parser.add_argument('--trigger-phase1', action='store_true', help='Trigger Phase 1 after Phase 0 completes (training mode only)')
    parser.add_argument('--phase1-job-id', type=int, default=980829916721201, help='Databricks Job ID for Phase 1')
    parser.add_argument('--n-trials', type=int, default=100, help='Number of Optuna trials for Phase 3 (training mode parameter)')
    
    # Inference mode triggers (Phase 0 → 6 → 5 → Cloudflare KV)
    parser.add_argument('--trigger-phase5', action='store_true', help='Trigger Phase 6→5 job chain after Phase 0 completes (requires --inference)')
    parser.add_argument('--phase5-job-id', type=int, default=899063372642667, help='Databricks Job ID for Phase 5 (Cloudflare KV version)')
    parser.add_argument('--phase6-job-id', type=int, default=366820514032698, help='Databricks Job ID for Phase 6')
    
    parser.add_argument('--selected-features-path', type=str, default='dbfs:/mnt/artifacts/selected_features_aug01_60.json', help='Path to selected features JSON file')
    parser.add_argument('--phase5-date-output', type=str, help='Output file path for extracted date (for job chaining). Default: /dbfs/mnt/vw_pipeline/artifacts/phase5_date.txt')
    # Optional: Model path to forward to Phase 6/5 (Inference Mode)
    parser.add_argument('--model-path', type=str, default=None, help='Path to VW model to forward to Phase 6/5 triggers (Inference Mode only)')
    # Optional: Skip training in Phase 6 (for initial deployment before model is in production)
    parser.add_argument('--skip-training', action='store_true', help='Skip model training in Phase 6 (only discover rewards and update decisions table). Use during initial deployment before model is serving in production.')

    # Optional: export a Pandas-friendly test parquet split for Phase 4
    parser.add_argument('--export-test-parquet', action='store_true', help='Export a user-hash test split as Parquet for Phase 4 Pandas-based validations')
    parser.add_argument('--test-parquet-path', type=str, default='', help='Output path for test_df.parquet (DBFS or local). Required if --export-test-parquet is set')
    parser.add_argument('--split-seed', type=int, default=42, help='Seed used for user-hash deterministic split when exporting test parquet')
    parser.add_argument('--test-ratio', type=float, default=0.1, help='Ratio of rows to include in test split when exporting Parquet (default 0.1)')
    args = parser.parse_args()

    # =========================================================================
    # LOAD PIPELINE CONFIG FROM JSON FILE
    # =========================================================================
    # This allows controlling all parameters from Databricks without modifying
    # Snowflake. Snowflake sends --date and --snowflake-table (dynamic per run).
    # All other params are loaded from config/pipeline_params.json
    # Command-line args always take precedence over config file.
    pipeline_config = load_pipeline_config()
    apply_pipeline_config(args, pipeline_config)

    # Validate n-trials
    if not 2 <= args.n_trials <= 300:
        parser.error(f"--n-trials must be between 2 and 300, got {args.n_trials}")
    
    if args.write_partitions <= 0:
        parser.error("--write-partitions must be a positive integer")

    if args.checkpoint_dir:
        delete_dbfs_path(args.checkpoint_dir)

    print("🚀 Starting Phase 0: Data Generation from Snowflake")
    print("=" * 80)
    if args.use_parquet:
        print(f"📂 DATA SOURCE: Parquet ({args.parquet_path})")
    elif args.use_csv:
        print(f"📂 DATA SOURCE: CSV ({args.csv_path})")
    else:
        print(f"📂 DATA SOURCE: Snowflake (gauserstartdate ≥ {args.start_date}) via {'Connector' if args.use_sf_connector else 'JDBC'}")
    print("=" * 80)

    # Validate and warn about local-dirs (shuffle spill)
    if args.local_dirs:
        local_dir_list = [d.strip() for d in args.local_dirs.split(',') if d.strip()]
        if '/tmp' in local_dir_list and len(local_dir_list) == 1:
            print("⚠️  WARNING: Using /tmp for shuffle spill. For large runs (4+ months), this may fill up.")
            print("   Consider using larger disk partitions: --local-dirs '/mnt/ssd1/spark,/mnt/ssd2/spark'")
            print("   Alternative: Use main disk: --local-dirs '$HOME/spark-temp' (create directory first)")
        # Check disk space for each directory and verify they exist
        for local_dir in local_dir_list:
            try:
                import os
                if not os.path.exists(local_dir):
                    print(f"⚠️  Local directory does not exist: {local_dir}")
                    print("   Attempting to create it...")
                    try:
                        os.makedirs(local_dir, exist_ok=True)
                        print(f"   ✅ Created {local_dir}")
                    except Exception as create_ex:
                        print(f"   ⚠️  Could not create {local_dir}: {create_ex}")
                if os.path.exists(local_dir):
                    statvfs = os.statvfs(local_dir)
                    free_gb = (statvfs.f_bavail * statvfs.f_frsize) / (1024**3)
                    if free_gb < 100:
                        print(f"⚠️  WARNING: {local_dir} has only {free_gb:.1f}GB free. May be insufficient for large shuffles.")
                        print(f"   Recommended: At least 200GB free for 2+ months of data")
                else:
                    print(f"⚠️  {local_dir} still does not exist. Spark will fall back to default temp dirs.")
            except Exception as e:
                print(f"⚠️  Could not check or create {local_dir}: {e}")
    else:
        print("⚠️  WARNING: No --local-dirs specified. Spark will use default temp directory (may be /tmp).")
        print("   For large runs, specify: --local-dirs '/path/to/large/disk/spark'")
    
    # Resume summary (UX)
    if args.resume_from != 'none':
        print("\n" + "=" * 80)
        print("🔄 RESUMING PIPELINE")
        print("=" * 80)
        print(f"   Checkpoint: {args.resume_from}")
        print(f"   Location:   {args.checkpoint_dir}")
        if args.resume_from == 'daily_bandit':
            print("   Skipping: Data loading, daily aggregation, logged actions, normalization/cleaning")
            print("   Running:  Advanced features → EWMA → Write")
        elif args.resume_from == 'daily_features':
            print("   Skipping: Data loading, daily aggregation")
            print("   Running:  Logged actions → Daily bandit → Normalization → Advanced → EWMA → Write")
        elif args.resume_from == 'advanced_features':
            print("   Skipping: Data loading, daily aggregation, logged actions, normalization, advanced features")
            print("   Running:  EWMA → Write")
        print("=" * 80 + "\n")

    # Combine packages: Delta Lake (if delta output) + Snowflake connector (if used)
    packages = []
    if args.output_format == 'delta':
        packages.append(args.delta_packages)
    if args.use_sf_connector and args.sf_packages:
        packages.append(args.sf_packages)
    combined_packages = ",".join(packages)
    
    spark = initialize_spark_session(
        shuffle_partitions=args.shuffle_partitions,
        local_dirs=args.local_dirs,
        enable_aqe=not args.disable_aqe,
        extra_packages=combined_packages,
        enable_delta_extensions=(args.output_format == 'delta'),
        force_local_mode=args.force_local_spark,
    )

    try:
        # 1) Load/resume
        resume = args.resume_from
        checkpoint_dir = normalize_path(args.checkpoint_dir)

        # Track original target date for inference mode (will be set later if --inference is used)
        inference_target_date = None
        
        if resume == 'daily_bandit':
            print("🔁 Resuming from checkpoint: daily_bandit")
            daily_bandit = read_checkpoint_dataset(spark, checkpoint_dir, 'daily_bandit', 'delta' if args.output_format=='delta' else 'parquet')
        else:
            distinct_session_dates = None

            if args.use_parquet:
                raw_events = load_from_parquet(spark, args.parquet_path)
            elif args.use_csv:
                raw_events = load_from_csv(spark, args.csv_path)
            else:
                # Validate start-date format (YYYY-MM-DD)
                try:
                    datetime.strptime(args.start_date, "%Y-%m-%d")
                except ValueError:
                    raise ValueError(f"Invalid --start-date '{args.start_date}'. Expected format YYYY-MM-DD.")

                # INFERENCE MODE: Extend date range for EWMA feature computation
                # EWMA features need historical data to compute meaningful values
                # Store original target date for filtering before final write
                inference_target_date = None
                if args.inference:
                    inference_target_date = args.start_date  # The date user wants predictions for
                    target_dt = datetime.strptime(args.start_date, "%Y-%m-%d")
                    lookback_dt = target_dt - timedelta(days=args.ewma_lookback_days)
                    lookback_start_date = lookback_dt.strftime("%Y-%m-%d")
                    print(f"📊 INFERENCE MODE: Extending date range for EWMA feature computation")
                    print(f"   Target date (predictions for): {inference_target_date}")
                    print(f"   EWMA lookback: {args.ewma_lookback_days} days")
                    print(f"   Query start date: {lookback_start_date} (extended from {args.start_date})")
                    args.start_date = lookback_start_date  # Extend query range
                
                # INFERENCE MODE OPTIMIZATION: Parse required features for selective computation
                # This enables skipping unnecessary feature computation for significant speedup
                required_features_info = None
                if args.inference and args.selected_features_path:
                    try:
                        required_features_info = parse_required_features(args.selected_features_path)
                    except Exception as e:
                        print(f"⚠️  Could not parse required features: {e}")
                        print(f"   Falling back to full feature computation")
                        required_features_info = None

                # Try loading JSON config if present
                json_cfg = load_snowflake_config_from_json(args.sf_config_json)

                if args.use_sf_connector:
                    # Build connector options with priority: CLI > JSON > in-script defaults
                    sf_url = args.sf_url or (json_cfg.get('url') if json_cfg and json_cfg.get('url') else '')
                    sf_user = args.sf_user or (json_cfg.get('user') if json_cfg and json_cfg.get('user') else '')
                    sf_password = args.sf_password or (json_cfg.get('password') if json_cfg and json_cfg.get('password') else '')
                    sf_account = args.sf_account or (json_cfg.get('account') if json_cfg and json_cfg.get('account') else '')
                    sf_warehouse = args.sf_warehouse or (json_cfg.get('warehouse') if json_cfg and json_cfg.get('warehouse') else 'ML_DATABRICKS')
                    sf_database = args.sf_database or (json_cfg.get('database') if json_cfg and json_cfg.get('database') else 'SPACEPLAY')
                    sf_schema = args.sf_schema or (json_cfg.get('schema') if json_cfg and json_cfg.get('schema') else 'UNITY')
                    sf_role = args.sf_role or (json_cfg.get('role') if json_cfg and json_cfg.get('role') else '')

                    # Fallback to hardcoded JDBC defaults if still missing
                    if not sf_user or not sf_password or (not sf_url and not sf_account):
                        _cfg, _ = connect_to_snowflake_direct()
                        sf_user = sf_user or _cfg.get('user')
                        sf_password = sf_password or _cfg.get('password')
                        sf_account = sf_account or _cfg.get('account')
                        sf_url = sf_url or (f"https://{sf_account}.snowflakecomputing.com" if sf_account else '')
                # Build Snowflake config with priority: CLI > JSON > in-script defaults
                snowflake_config = {
                    "user": args.sf_user or (json_cfg.get('user') if json_cfg else ''),
                    "password": args.sf_password or (json_cfg.get('password') if json_cfg else ''),
                    "account": args.sf_account or (json_cfg.get('account') if json_cfg else ''),
                    "warehouse": args.sf_warehouse or (json_cfg.get('warehouse') if json_cfg else 'ML_DATABRICKS'),
                    "database": args.sf_database or (json_cfg.get('database') if json_cfg else 'SPACEPLAY'),
                    "schema": args.sf_schema or (json_cfg.get('schema') if json_cfg else 'UNITY'),
                    "role": args.sf_role or (json_cfg.get('role') if json_cfg else ''),
                    "url": args.sf_url or (json_cfg.get('url') if json_cfg else ''),
                }

                # Fallback to hardcoded JDBC defaults if still missing
                if not snowflake_config['user'] or not snowflake_config['password'] or (not snowflake_config['url'] and not snowflake_config['account']):
                    _cfg, _ = connect_to_snowflake_direct()
                    snowflake_config['user'] = snowflake_config['user'] or _cfg.get('user')
                    snowflake_config['password'] = snowflake_config['password'] or _cfg.get('password')
                    snowflake_config['account'] = snowflake_config['account'] or _cfg.get('account')
                    snowflake_config['url'] = snowflake_config['url'] or (f"https://{snowflake_config['account']}.snowflakecomputing.com" if snowflake_config['account'] else '')
                    snowflake_config['warehouse'] = snowflake_config['warehouse'] or _cfg.get('warehouse')
                    snowflake_config['database'] = snowflake_config['database'] or _cfg.get('database')
                    snowflake_config['schema'] = snowflake_config['schema'] or _cfg.get('schema')

                if not snowflake_config['url'] or not snowflake_config['user'] or not snowflake_config['password']:
                    raise ValueError("Missing Snowflake connector credentials. Provide CLI/JSON or rely on hardcoded defaults.")

                # Determine JDBC URL for non-connector path
                jdbc_url = f"jdbc:snowflake://{snowflake_config['account']}.snowflakecomputing.com/"

                if args.use_parquet:
                    raw_events = load_from_parquet(spark, args.parquet_path)
                elif args.use_csv:
                    raw_events = load_from_csv(spark, args.csv_path)
                elif args.use_sf_connector:
                    # Spark Snowflake Connector
                    sf_options = {
                        "sfUrl": snowflake_config["url"],
                        "sfUser": snowflake_config["user"],
                        "sfPassword": snowflake_config["password"],
                        "sfDatabase": snowflake_config["database"],
                        "sfSchema": snowflake_config["schema"],
                        "sfWarehouse": snowflake_config["warehouse"],
                    }
                    # Add role if present
                    if snowflake_config.get("role"):
                        sf_options["sfRole"] = snowflake_config["role"]
                        
                    raw_events = create_snowflake_connector_dataframe(spark, sf_options, start_date=args.start_date, end_date=args.end_date, snowflake_table=args.snowflake_table)
                else:
                    # JDBC fallback
                    raw_events = create_spark_dataframe(spark, snowflake_config, jdbc_url, start_date=args.start_date, end_date=args.end_date, snowflake_table=args.snowflake_table)

        if resume == 'daily_bandit':
            pass
        else:
            # 2) Prepare dates and filter to multi-day users
            # Pass start/end date to strictly filter events (since SQL filter is on install date)
            events_with_date = add_session_date_column(raw_events, args.start_date, args.end_date)

            # DEBUG: Print min/max dates to verify data range
            print("\n🔍 DEBUG: Verifying date ranges...")
            try:
                date_stats = events_with_date.agg(
                    F.min("session_date").alias("min_session"),
                    F.max("session_date").alias("max_session"),
                    F.min("gauserstartdate").alias("min_install"),
                    F.max("gauserstartdate").alias("max_install")
                ).collect()[0]
                print(f"   📅 Session Date Range: {date_stats['min_session']} to {date_stats['max_session']}")
                print(f"   👶 Install Date Range: {date_stats['min_install']} to {date_stats['max_install']}")
            except Exception as e:
                print(f"   ⚠️  Could not compute date stats: {e}")
            print("=" * 40 + "\n")

            # Count distinct session dates early for downstream partition planning
            try:
                distinct_session_dates = events_with_date.select("session_date").distinct().count()
                print(f"📆 Distinct session_date count in raw data: {distinct_session_dates:,}")
            except Exception as e:
                print(f"⚠️  Could not count distinct session_date values: {e}")
                distinct_session_dates = None
            # Skip multi-day filter for inference (single-day data would filter out all users)
            if args.inference:
                print("⚠️  Inference mode: Skipping multi-day user filter (processing single-day data)")
                multi_day_events = events_with_date
            else:
                multi_day_events = retain_multi_day_users(events_with_date)

            # 3) Logged actions and arms
            logged_actions = extract_logged_action(multi_day_events)
            arm_assignments = assign_difficulty_arms(logged_actions, inference_mode=args.inference)

            if resume == 'daily_features':
                print("🔁 Resuming from daily_features checkpoint")
                daily_features = read_checkpoint_dataset(spark, checkpoint_dir, 'daily_features', 'delta' if args.output_format=='delta' else 'parquet')
            else:
                # 4) Aggregate core daily features
                # In inference mode, pass required features for optimization
                agg_required_features = None
                if required_features_info and args.inference:
                    agg_required_features = required_features_info.get('required_base', set())
                daily_features = aggregate_daily_features(multi_day_events, required_features=agg_required_features)
                if args.enable_checkpoints:
                    write_checkpoint_dataset(spark, daily_features, checkpoint_dir, 'daily_features', partition_col=args.partition_column, fmt=args.output_format)

            # 5) Create bandit dataset (features + action + reward)
            daily_bandit = create_daily_bandit_dataset(
                daily_features,
                logged_actions,
                arm_assignments,
                inference_mode=args.inference
            )

        # 6) Normalize columns, clean low-variance columns (Spark-only)
        if resume != 'daily_bandit':
            daily_bandit = normalize_to_user_lowercase(daily_bandit)
            daily_bandit = clean_null_and_zero_columns_spark(daily_bandit, id_cols=["user_id", "session_date", "difficulty_arm"])
            # FORCE MATERIALIZATION: Write and immediately read back to break Snowflake connector lineage
            # This prevents the connector from being accessed during advanced features computation
            if args.enable_checkpoints:
                write_checkpoint_dataset(spark, daily_bandit, checkpoint_dir, 'daily_bandit', partition_col=args.partition_column, fmt=args.output_format)
                print("💾 Materializing daily_bandit to break Spark lineage from Snowflake connector...")
                daily_bandit = read_checkpoint_dataset(spark, checkpoint_dir, 'daily_bandit', 'delta' if args.output_format=='delta' else 'parquet')
            elif args.use_sf_connector and not args.use_parquet and not args.use_csv:
                # Even without explicit checkpoints, force materialization when using Snowflake connector
                # to avoid telemetry issues during advanced features
                print("💾 Auto-materializing daily_bandit to break Snowflake connector lineage...")
                temp_checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir else 'dbfs:/mnt/bandit/checkpoints'
                # normalize_path will handle absolute path conversion
                temp_checkpoint_dir = normalize_path(temp_checkpoint_dir)
                write_checkpoint_dataset(spark, daily_bandit, temp_checkpoint_dir, 'daily_bandit', partition_col=args.partition_column, fmt=args.output_format)
                daily_bandit = read_checkpoint_dataset(spark, temp_checkpoint_dir, 'daily_bandit', 'delta' if args.output_format=='delta' else 'parquet')
        elif args.enable_checkpoints:
            # Resuming: already materialized
            pass

        # 7) Advanced gameplay features (Spark windows)
        if resume == 'advanced_features':
            print("🔁 Resuming from checkpoint: advanced_features")
            daily_with_adv = read_checkpoint_dataset(spark, checkpoint_dir, 'advanced_features', 'delta' if args.output_format=='delta' else 'parquet')
        else:
            # CRITICAL: Repartition by user_id BEFORE window functions
            # Window functions with partitionBy("user_id") require all user data on same partition
            # Pre-partitioning prevents full shuffle and ensures proper distribution
            # NOTE: Using fixed partition count to avoid .count() operations that materialize data
            print("🔄 Repartitioning by user_id for window functions...")
            try:
                # Use fixed partition count - avoids expensive count() operations that materialize data
                # 100 partitions works well for:
                # - 10K-1M users: ~100-10K users per partition (balanced)
                # - Prevents data skew (no single partition too large)
                # - Good parallelism for window functions
                fixed_partitions = 100
                print(f"   Using fixed partition count: {fixed_partitions}")
                print(f"   Target: ~1000-2000 users per partition (optimal for window functions)")
                daily_bandit = daily_bandit.repartition(fixed_partitions, "user_id")
                print(f"   ✅ Repartitioned to {fixed_partitions} partitions")
            except Exception as e:
                print(f"❌ Repartition failed: {e}")
                print("   Continuing without repartition - may cause memory issues during window functions")
            
            # In inference mode, pass required features for optimization
            adv_feature_whitelist = None
            if required_features_info and args.inference:
                # Combine required_advanced and any EWMA base columns that are advanced features
                adv_feature_whitelist = required_features_info.get('required_advanced', set())
                # Also include features that EWMA depends on
                ewma_base_cols = set(required_features_info.get('ewma_specs', {}).keys())
                for col in ewma_base_cols:
                    if col in ADVANCED_FEATURE_DEPS:
                        adv_feature_whitelist.add(col)
            
            daily_with_adv = add_advanced_gameplay_features_spark(daily_bandit, feature_whitelist=adv_feature_whitelist)
            
            if args.enable_checkpoints and resume not in ('daily_bandit','advanced_features') and not args.skip_advanced_materialization:
                print("💾 Materializing advanced features via checkpoint...")
                write_checkpoint_dataset(spark, daily_with_adv, checkpoint_dir, 'advanced_features', partition_col=args.partition_column, fmt=args.output_format)
                daily_with_adv = read_checkpoint_dataset(spark, checkpoint_dir, 'advanced_features', 'delta' if args.output_format=='delta' else 'parquet')
            elif args.skip_advanced_materialization:
                print("⚠️  Skipping advanced features materialization (--skip-advanced-materialization)")
            else:
                print("💾 Materializing advanced features in-cluster (persist + count)...")
                storage_level_names = ['MEMORY_ONLY','MEMORY_AND_DISK','DISK_ONLY','MEMORY_ONLY_SER','MEMORY_AND_DISK_SER','DISK_ONLY_2','NONE']
                storage_level_map = {}
                for name in storage_level_names:
                    level = getattr(StorageLevel, name, None)
                    if level is not None:
                        storage_level_map[name] = level
                mat_level = storage_level_map.get(args.materialize_storage_level.upper(), StorageLevel.DISK_ONLY)
                target_mat_partitions = max(args.materialize_partitions, 50)
                # Repartition directly without checking current count (avoids materialization)
                print(f"   Repartitioning advanced features to {target_mat_partitions} before persist...")
                daily_with_adv = daily_with_adv.repartition(target_mat_partitions, "user_id")
                daily_with_adv = daily_with_adv.persist(mat_level)
                mat_start = datetime.utcnow()
                row_count = daily_with_adv.count()
                print(f"✅ Advanced features materialized via persist/count ({row_count:,} rows) in {(datetime.utcnow()-mat_start).total_seconds():.1f}s")

        # 8) EWMA features via grouped Pandas UDF (fast and vectorized inside Spark)
        if args.disable_ewma:
            print("⏭️  Skipping EWMA features (--disable-ewma set)")
        else:
            ewma_cols_list = [c.strip() for c in args.ewma_cols.split(',')] if args.ewma_cols else None
            
            # INFERENCE MODE OPTIMIZATION: Only compute EWMA features that are actually selected
            ewma_filter = None
            if args.inference and args.selected_features_path:
                print(f"\n🔧 INFERENCE MODE: Loading selected features for EWMA optimization...")
                # Use ewma_specs from required_features_info if already computed
                if required_features_info and required_features_info.get('ewma_specs'):
                    ewma_filter = required_features_info['ewma_specs']
                    print(f"   Using pre-computed EWMA specs: {len(ewma_filter)} base columns")
                else:
                    ewma_filter = parse_ewma_from_selected_features(args.selected_features_path)
                if not ewma_filter:
                    print(f"   ⚠️  Could not parse EWMA filter, computing all EWMA features")
            
            # DEBUG: Check dates BEFORE EWMA
            if args.inference:
                pre_ewma_dates = daily_with_adv.select(args.partition_column).distinct().collect()
                pre_ewma_date_strs = sorted([str(r[0]) for r in pre_ewma_dates])
                print(f"   🔍 DEBUG: Dates BEFORE EWMA ({len(pre_ewma_date_strs)}): {pre_ewma_date_strs[-5:]} (last 5)")
            
            daily_with_adv = compute_ewma_features_spark(
                daily_with_adv, 
                ewma_cols=ewma_cols_list,
                ewma_filter=ewma_filter
            )
            
            # DEBUG: Check dates AFTER EWMA
            if args.inference:
                post_ewma_dates = daily_with_adv.select(args.partition_column).distinct().collect()
                post_ewma_date_strs = sorted([str(r[0]) for r in post_ewma_dates])
                print(f"   🔍 DEBUG: Dates AFTER EWMA ({len(post_ewma_date_strs)}): {post_ewma_date_strs[-5:]} (last 5)")
                if len(pre_ewma_date_strs) != len(post_ewma_date_strs):
                    print(f"   ⚠️  WARNING: Date count changed! Before={len(pre_ewma_date_strs)}, After={len(post_ewma_date_strs)}")
                    missing = set(pre_ewma_date_strs) - set(post_ewma_date_strs)
                    if missing:
                        print(f"   ⚠️  MISSING DATES: {missing}")

        # 9) Lowercase all columns for consistency
        daily_with_adv = lowercase_all_columns(daily_with_adv)
        
        # 9b) INFERENCE MODE: Filter to target date BEFORE expensive checkpoint writes
        # EWMA needed all 31 days for computation, but we only want to write the target date
        if args.inference and inference_target_date:
            print(f"\n📅 INFERENCE MODE: Filtering to target date BEFORE checkpoint writes")
            print(f"   Target date: {inference_target_date}")
            pre_filter_dates = daily_with_adv.select(args.partition_column).distinct().count()
            print(f"   Dates before filter: {pre_filter_dates}")
            daily_with_adv = daily_with_adv.filter(F.col(args.partition_column) == F.lit(inference_target_date))
            post_filter_dates = daily_with_adv.select(args.partition_column).distinct().count()
            post_filter_rows = daily_with_adv.count()
            print(f"   Dates after filter: {post_filter_dates}")
            print(f"   Rows after filter: {post_filter_rows:,}")
            if post_filter_rows == 0:
                raise ValueError(f"No data found for target date {inference_target_date} after EWMA computation. "
                               f"Check if the target date exists in the source data.")
            print(f"   ✅ Filtered to target date - checkpoint writes will be much faster!")
        
        # 9c) INFERENCE MODE: Select only columns needed for inference (from selected_features.json)
        # This reduces storage and speeds up downstream reads
        if args.inference and args.selected_features_path:
            print(f"\n📋 INFERENCE MODE: Selecting only features from selected_features.json")
            try:
                # Load selected features
                sf_path = args.selected_features_path
                if sf_path.startswith("dbfs:/"):
                    sf_local_path = "/dbfs" + sf_path[5:]
                else:
                    sf_local_path = sf_path
                
                with open(sf_local_path, 'r') as f:
                    sf_data = json.load(f)
                selected_features = sf_data.get('selected_features', [])
                
                # Required columns that must always be included
                required_cols = {
                    'user_id', 'session_date', 
                    # Include logged action info for debugging/analysis
                    'current_effectivelevelmultiplier', 'action', 'difficulty_arm',
                    'next_effectivelevelmultiplier', 'previous_day_action', 'previous_day_multiplier'
                }
                
                # Build final column list
                available_cols = set(daily_with_adv.columns)
                available_cols_lower = {c.lower(): c for c in available_cols}
                
                # Start with required columns
                final_cols = []
                for req_col in required_cols:
                    if req_col in available_cols:
                        final_cols.append(req_col)
                    elif req_col.lower() in available_cols_lower:
                        final_cols.append(available_cols_lower[req_col.lower()])
                
                # Add selected features (case-insensitive matching)
                selected_found = 0
                selected_missing = []
                for feat in selected_features:
                    feat_lower = feat.lower()
                    if feat_lower in available_cols_lower:
                        actual_col = available_cols_lower[feat_lower]
                        if actual_col not in final_cols:
                            final_cols.append(actual_col)
                        selected_found += 1
                    else:
                        selected_missing.append(feat)
                
                # Also add partition column if not already included
                if args.partition_column not in final_cols:
                    if args.partition_column in available_cols:
                        final_cols.append(args.partition_column)
                    elif args.partition_column.lower() in available_cols_lower:
                        final_cols.append(available_cols_lower[args.partition_column.lower()])
                
                pre_select_cols = len(daily_with_adv.columns)
                daily_with_adv = daily_with_adv.select(*final_cols)
                post_select_cols = len(daily_with_adv.columns)
                
                print(f"   Loaded {len(selected_features)} features from {args.selected_features_path}")
                print(f"   Found {selected_found}/{len(selected_features)} selected features in DataFrame")
                print(f"   Columns: {pre_select_cols} → {post_select_cols} ({pre_select_cols - post_select_cols} dropped)")
                
                if selected_missing:
                    print(f"   ⚠️  {len(selected_missing)} selected features not found: {selected_missing[:5]}...")
                
                print(f"   ✅ Column selection complete - only writing necessary features!")
                
            except Exception as e:
                print(f"   ⚠️  Could not load selected features for column selection: {e}")
                print(f"   Continuing with all columns...")
        
        # Materialization before final write: Skip if memory is constrained
        # Write operations can be memory-intensive, but materialization can also cause OOM
        # For large datasets, it's often better to write directly without materialization
        print("💾 Preparing final dataset for write...")
        
        # Check if we should skip materialization (based on executor memory or flag)
        skip_materialization = args.skip_final_materialization
        if not skip_materialization:
            try:
                executor_memory_gb = int(spark.conf.get("spark.executor.memory", "6g").replace("g", ""))
                skip_materialization = executor_memory_gb < 8  # Skip if less than 8GB executor memory
            except:
                skip_materialization = True  # Default to skip if we can't determine
        
        if skip_materialization:
            print("   Skipping materialization (low memory cluster) - writing directly...")
            print("   Note: Direct write may be slower but avoids OOM")
        else:
            print("   Attempting materialization before write...")
            temp_final_dir = normalize_path('dbfs:/mnt/bandit/temp/final_dataset_mat')
            try:
                dbutils
                driver_temp_dir = to_driver_dbfs_path(temp_final_dir)
                dbfs_path = driver_temp_dir.replace('/dbfs/', '/') if driver_temp_dir.startswith('/dbfs/') else driver_temp_dir
                dbutils.fs.mkdirs(dbfs_path)
            except:
                pass
            
            # Materialize with error handling for Delta exception handler issues
            try:
                # More aggressive coalescing for materialization
                # Coalesce directly without checking partition count (avoids materialization)
                print(f"   Repartitioning to 50 partitions for materialization...")
                daily_with_adv = daily_with_adv.repartition(50, F.col(args.partition_column))
                
                spark_temp_final = to_spark_path(temp_final_dir)
                daily_with_adv.write.mode('overwrite').parquet(spark_temp_final)
                daily_with_adv = spark.read.parquet(spark_temp_final)
                print("✅ Final dataset materialized")
            except Exception as mat_error:
                error_msg = str(mat_error)
                if "delta.exceptions.captured" in error_msg or "ModuleNotFoundError" in error_msg:
                    print(f"⚠️  Final materialization hit Delta exception handler issue")
                    print("   Skipping materialization - proceeding with direct write")
                    skip_materialization = True
                elif "DriverStoppedException" in error_msg or "exit code: 134" in error_msg or "OOM" in error_msg:
                    print(f"⚠️  Final materialization caused OOM: {error_msg}")
                    print("   Skipping materialization - proceeding with direct write")
                    skip_materialization = True
                else:
                    print(f"⚠️  Materialization failed: {error_msg}")
                    print("   Skipping materialization - proceeding with direct write")
                    skip_materialization = True
        
        # Materialize final dataset via Delta checkpoint to break complex lineage (unless --skip-checkpoint)
        if not args.skip_checkpoint:
            print("💾 Materializing final dataset via Delta checkpoint before write...")
            final_checkpoint_dir = normalize_path('dbfs:/mnt/bandit/temp/final_write_checkpoint')
            write_checkpoint_dataset(
                spark,
                daily_with_adv,
                final_checkpoint_dir,
                'final_write',
                partition_col=args.partition_column,
                fmt='delta'
            )
            daily_with_adv = read_checkpoint_dataset(spark, final_checkpoint_dir, 'final_write', 'delta')
        else:
            print("⏭️  Skipping final checkpoint (--skip-checkpoint set)")

        # If materialization was skipped or failed earlier, ensure adequate parallelism before direct write
        if skip_materialization:
            target_partitions = max(args.write_partitions, 200)
            # More partitions = smaller partitions = less memory per partition = less OOM risk
            # Repartitioning triggers lineage execution, but checkpoint write will execute it anyway
            # So we want MORE partitions to reduce memory per partition during checkpoint write
            print(f"   Repartitioning to {target_partitions} partitions before direct write...")
            print(f"   More partitions = smaller partitions = less memory per task = less OOM risk")
            daily_with_adv = daily_with_adv.repartition(target_partitions, F.col(args.partition_column))
            
            # Add checkpoint to break lineage before write (reduces memory pressure during final write)
            # This writes to disk once, then reads back, which simplifies the execution plan
            # Even without Photon, this helps by breaking complex window function lineage chains
            # Can be skipped with --skip-checkpoint flag
            # Detect cluster size to tune partition counts for checkpoint/direct write
            try:
                executor_count = int(spark.conf.get("spark.executor.instances", "1"))
                if executor_count == 0:
                    executor_count = len(spark.sparkContext.statusTracker().getExecutorInfos())
                if executor_count == 0:
                    executor_count = 1
            except:
                executor_count = 1
            try:
                cores_per_executor = int(spark.conf.get("spark.executor.cores", "4"))
            except:
                cores_per_executor = 4
            total_cores = executor_count * cores_per_executor
            cluster_partition_cap = 1_000_000_000  # effectively uncapped
            max_partitions_by_cluster = max(200, cluster_partition_cap)

            if args.skip_checkpoint:
                print("   ⚠️  Skipping checkpoint (--skip-checkpoint) - writing directly to Delta with high partition count")
                print("   Note: Direct write may be slower but avoids the extra checkpoint write")
                print(f"   Cluster size: {executor_count} executors × {cores_per_executor} cores = {total_cores} total cores")
                print(f"   Cluster cap: {max_partitions_by_cluster} partitions (global safety cap)")

                if args.final_write_partitions > 0:
                    desired_final_partitions = args.final_write_partitions
                    print(f"   Using user-provided final partitions: {desired_final_partitions}")
                else:
                    print(f"   Estimating required partitions using {args.partitions_per_day:,} partitions per distinct session date...")
                    try:
                        day_count = daily_with_adv.select(args.partition_column).distinct().count()
                    except Exception as day_err:
                        print(f"   ⚠️  Could not count distinct {args.partition_column}: {day_err}")
                        day_count = 1
                    desired_final_partitions = max(target_partitions, args.partitions_per_day * max(1, day_count))
                    print(f"   Distinct {args.partition_column} values: {day_count:,}")
                    print(f"   Planned partitions (per-day heuristic): {desired_final_partitions:,}")

                final_partitions = min(max(desired_final_partitions, target_partitions), max_partitions_by_cluster)
                print(f"   Using {final_partitions} partitions for final Delta write (smaller partitions → lower memory per task)")
                if final_partitions != target_partitions:
                    daily_with_adv = daily_with_adv.repartition(final_partitions, F.col(args.partition_column))
                args.write_partitions = final_partitions
            else:
                print(f"   Adding checkpoint to break lineage before write (reduces memory pressure)...")
                desired_checkpoint_partitions = max(target_partitions * 2, 400)
                checkpoint_partitions = min(desired_checkpoint_partitions, max_partitions_by_cluster)
                print(f"   Cluster size: {executor_count} executors × {cores_per_executor} cores = {total_cores} total cores")
                print(f"   Desired checkpoint partitions: {desired_checkpoint_partitions} (2x original {target_partitions})")
                print(f"   Cluster limit: {max_partitions_by_cluster} (global safety cap)")
                print(f"   Using {checkpoint_partitions} partitions for checkpoint write...")
                print(f"   More partitions = smaller partitions = less memory per task during checkpoint write")
                daily_with_adv = daily_with_adv.repartition(checkpoint_partitions, F.col(args.partition_column))

                checkpoint_dir = normalize_path('dbfs:/mnt/bandit/temp/write_checkpoint')
                try:
                    try:
                        dbutils
                        driver_checkpoint_dir = to_driver_dbfs_path(checkpoint_dir)
                        dbfs_path = driver_checkpoint_dir.replace('/dbfs/', '/') if driver_checkpoint_dir.startswith('/dbfs/') else driver_checkpoint_dir
                        dbutils.fs.rm(dbfs_path, True)
                    except:
                        import os
                        import shutil
                        if os.path.exists(checkpoint_dir):
                            shutil.rmtree(checkpoint_dir)

                    print(f"   Writing checkpoint to {checkpoint_dir} (this may take a while)...")
                    spark_checkpoint_dir = to_spark_path(checkpoint_dir)
                    daily_with_adv.write.mode('overwrite').parquet(spark_checkpoint_dir)

                    print(f"   Reading checkpoint back...")
                    daily_with_adv = spark.read.parquet(spark_checkpoint_dir)

                    desired_partitions = max(target_partitions * 3, 4000)
                    final_write_partitions = min(desired_partitions, max_partitions_by_cluster)
                    print(f"   Desired partitions: {desired_partitions} (3x original {target_partitions})")
                    print(f"   Cluster limit: {max_partitions_by_cluster} (global safety cap)")
                    print(f"   Using {final_write_partitions} partitions for final Delta write...")
                    print(f"   More partitions = smaller partitions = less memory per task during Delta write")
                    daily_with_adv = daily_with_adv.repartition(final_write_partitions, F.col(args.partition_column))
                    args.write_partitions = final_write_partitions

                    print("   ✅ Checkpoint complete - lineage broken, ready for final write")
                except Exception as checkpoint_error:
                    error_msg = str(checkpoint_error)
                    if "OOM" in error_msg or "out of memory" in error_msg.lower() or "SparkStoppedException" in error_msg:
                        print(f"   ⚠️  Checkpoint failed due to OOM/Spark crash: {error_msg[:200]}")
                        print("   The checkpoint write itself triggered OOM - cluster may be too small")
                        print("   Options:")
                        print("     1. Increase cluster memory")
                        print("     2. Process smaller date ranges")
                        print("     3. Use --skip-checkpoint (risky, may still OOM on final write)")
                        print("   Proceeding with direct write (may still fail with OOM)")
                    else:
                        print(f"   ⚠️  Checkpoint failed: {error_msg[:200]}")
                        print("   Proceeding with direct write")

        
        # Get the processing date for Phase 5 triggering (use original target date, not extended lookback)
        processing_date = None  # For Phase 5 job chaining
        if args.inference:
            # Use the ORIGINAL target date (not the extended lookback start_date)
            if inference_target_date:
                processing_date = inference_target_date
            elif args.replace_partition_date:
                processing_date = args.replace_partition_date
            else:
                from datetime import datetime as dt
                processing_date = dt.now().strftime("%Y-%m-%d")
            print(f"   📅 Processing date for Phase 5: {processing_date}")
        
        # 10) Write dataset partitioned by date only (recommended to reduce small files)
        # For daily runs, use replaceWhere to update only that day's partition
        replace_date = None
        
        if args.write_mode == "replaceWhere":
            if inference_target_date:
                # Use original target date for partition replacement (not extended lookback)
                replace_date = inference_target_date
                print(f"   Using inference target date for partition replacement: {replace_date}")
            elif args.replace_partition_date:
                replace_date = args.replace_partition_date
            elif args.start_date:
                replace_date = args.start_date
                print(f"   Using start_date for partition replacement: {replace_date}")
            else:
                # In inference or replaceWhere mode we require an explicit date parameter
                raise ValueError("Cannot determine partition date for replaceWhere. Provide --start-date or --replace-partition-date.")
        
        write_partitioned_dataset(
            spark,
            daily_with_adv,
            args.out_dir,
            partition_col=args.partition_column,
            fmt=args.output_format,
            target_partitions=args.write_partitions,
            max_records_per_file=args.max_records_per_file,
            write_mode=args.write_mode,
            replace_partition_date=replace_date,
            replace_partition_end_date=args.end_date
        )

        # Optional: Export a small Pandas-friendly test parquet for Phase 4
        if args.export_test_parquet:
            if not args.test_parquet_path:
                print("⚠️  --export-test-parquet was set but --test-parquet-path is empty; skipping test parquet export")
            else:
                try:
                    print("\n" + "=" * 80)
                    print("EXPORTING TEST PARQUET (Phase 4 helper)")
                    print("=" * 80)
                    # Deterministic user-based split (same strategy used elsewhere)
                    # Use normalized lower-case user_id (present after lowercase_all_columns)
                    df_for_split = daily_with_adv
                    if 'user_id' not in df_for_split.columns:
                        # Best-effort: attempt to alias USERID if present
                        if 'USERID' in df_for_split.columns:
                            df_for_split = df_for_split.withColumnRenamed('USERID', 'user_id')
                        else:
                            raise ValueError("Expected 'user_id' column for split; not found")
                    # Bucket 0..99
                    bucket = F.pmod(F.abs(F.hash(F.col('user_id')) + F.lit(args.split_seed)), F.lit(100))
                    df_buck = df_for_split.withColumn('__bucket__', bucket.cast('int'))
                    # Test threshold
                    test_floor = int(round((1.0 - float(args.test_ratio)) * 100))
                    test_floor = min(max(test_floor, 0), 99)
                    print(f"  Using test_floor bucket >= {test_floor} (ratio ~{args.test_ratio:.2f}, seed={args.split_seed})")
                    test_df = df_buck.filter(F.col('__bucket__') >= F.lit(test_floor)).drop('__bucket__')
                    out_test_path = normalize_path(args.test_parquet_path)
                    print(f"  Writing test parquet to: {out_test_path}")
                    (test_df
                        .repartition(64)  # keep it reasonably small
                        .write
                        .mode('overwrite')
                        .parquet(to_spark_path(out_test_path)))
                    print("✅ Test parquet export complete")
                except Exception as exp:
                    print(f"⚠️  Test parquet export failed: {exp}")

        print("\n" + "=" * 80)
        print("📊 FINAL SUMMARY (PURE SPARK)")
        
        # Training mode: Optionally trigger Phase 1
        if args.trigger_phase1 and not args.inference:
            print("\n" + "=" * 80)
            print("TRAINING MODE: PHASE 1 JOB CHAINING")
            print("=" * 80)
            
            try:
                import requests
                
                # Get Databricks token
                databricks_token = None
                try:
                    from pyspark.dbutils import DBUtils
                    dbutils = DBUtils()
                    try:
                        databricks_token = dbutils.secrets.get(scope="databricks", key="token")
                    except Exception:
                        databricks_token = os.environ.get("DATABRICKS_TOKEN")
                except Exception:
                    databricks_token = os.environ.get("DATABRICKS_TOKEN")
                
                if not databricks_token:
                    print("   ⚠️  No Databricks token found; skipping Phase 1 trigger")
                else:
                    # Get workspace URL
                    workspace_url = os.environ.get("DATABRICKS_WORKSPACE_URL", "https://adb-249008710733422.2.azuredatabricks.net")
                    if not workspace_url.startswith("http"):
                        workspace_url = f"https://{workspace_url}"
                    
                    # Trigger Phase 1 job
                    api_url = f"{workspace_url}/api/2.1/jobs/run-now"
                    
                    # Build parameters - only include end-date if specified
                    python_params = [
                        "--dataset-path", args.out_dir,
                        "--start-date", args.start_date,
                    ]
                    
                    # Only add end-date if it was specified (don't default to start-date)
                    if args.end_date:
                        python_params.extend(["--end-date", args.end_date])
                    
                    python_params.extend([
                        "--n-features", "60",
                        "--output-selected-features", "dbfs:/mnt/artifacts/selected_features_60.json",
                        "--train-propensity",
                        "--propensity-model-out", "dbfs:/mnt/models/propensity_spark",
                        "--n-trials", str(args.n_trials)  # Pass through to Phase 3
                    ])
                    
                    payload = {
                        "job_id": args.phase1_job_id,
                        "python_params": python_params
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
                            
                            print(f"   ✅ Triggered Phase 1 (Job ID: {args.phase1_job_id}, Run ID: {run_id})")
                            logging.info(f"Triggered Phase 1 (Job ID: {args.phase1_job_id}, Run ID: {run_id})")
                            print(f"   🔗 View run: {workspace_url}/#job/{args.phase1_job_id}/run/{run_id}")
                            print(f"   📊 Training Pipeline: Phase 0 → Phase 1 (Feature Selection)")
                            break  # Success, exit retry loop
                            
                        except Exception as retry_error:
                            if attempt < max_retries:
                                print(f"   🔄 Retry {attempt + 1}/{max_retries} after error: {retry_error}")
                                logging.warning(f"Phase 1 trigger attempt {attempt + 1} failed: {retry_error}")
                                print(f"   ⏳ Waiting 5 seconds before retry...")
                                import time
                                time.sleep(5)
                            else:
                                print(f"   ❌ Failed to trigger Phase 1 after {max_retries + 1} attempts: {retry_error}")
                                logging.error(f"Phase 1 trigger failed after {max_retries + 1} attempts: {retry_error}")
                    
            except Exception as e:
                print(f"   ⚠️  Failed to trigger Phase 1: {e}")
                import traceback
                traceback.print_exc()
        
        # Inference mode: Save processing date and optionally trigger Phase 5
        if args.inference and processing_date:
            print("\n" + "=" * 80)
            print("INFERENCE MODE: PHASE 5 JOB CHAINING")
            print("=" * 80)
            
            # Save date to file for job chaining
            date_output_path = args.phase5_date_output or "/dbfs/mnt/vw_pipeline/artifacts/phase5_date.txt"
            try:
                # Ensure directory exists
                date_output_dir = os.path.dirname(date_output_path)
                if date_output_dir.startswith("/dbfs/"):
                    try:
                        dbutils  # Check if in Databricks
                        dbfs_dir = date_output_dir.replace("/dbfs/", "/")
                        dbutils.fs.mkdirs(dbfs_dir)
                    except NameError:
                        os.makedirs(date_output_dir, exist_ok=True)
                else:
                    os.makedirs(date_output_dir, exist_ok=True)
                
                # Write date to file
                with open(date_output_path, 'w') as f:
                    f.write(processing_date)
                print(f"   ✅ Saved processing date to: {date_output_path}")
                print(f"   📅 Date for Phase 5: {processing_date}")
            except Exception as e:
                print(f"   ⚠️  Failed to save date to file: {e}")
            
            # Optionally trigger Phase 6 (Online Learning) then Phase 5 (Batch Inference)
            if args.trigger_phase5:
                if args.inference:
                    # INFERENCE MODE: Trigger Phase 6 first (which will trigger Phase 5)
                    if not args.phase6_job_id:
                        print("   ⚠️  --trigger-phase5 in inference mode requires --phase6-job-id; falling back to direct Phase 5 trigger")
                        trigger_job_id = args.phase5_job_id
                        job_name = "Phase 5 (Direct)"
                    else:
                        trigger_job_id = args.phase6_job_id
                        job_name = "Phase 6 (Online Learning → Phase 5)"
                else:
                    # TRAINING MODE: Do NOT trigger Phase 5/6
                    # Training flow: Phase 0 → Phase 2 → Phase 3 → Phase 4 (manual)
                    print("   ⚠️  --trigger-phase5 requires --inference mode")
                    print("   📚 Training mode: Phase 0 → Phase 2 → Phase 3 → Phase 4 (manual)")
                    print("   🚀 Inference mode: Phase 0 → Phase 6 → Phase 5 → Redis (automated)")
                    trigger_job_id = None  # Block trigger
                
                if not trigger_job_id:
                    print(f"   ⚠️  No job ID configured; skipping trigger")
                else:
                    try:
                        import requests
                        # Get Databricks token
                        databricks_token = None
                        try:
                            from pyspark.dbutils import DBUtils
                            dbutils = DBUtils()
                            try:
                                databricks_token = dbutils.secrets.get(scope="databricks", key="token")
                            except Exception:
                                # Try environment variable
                                databricks_token = os.environ.get("DATABRICKS_TOKEN")
                        except Exception:
                            databricks_token = os.environ.get("DATABRICKS_TOKEN")
                        
                        if not databricks_token:
                            print("   ⚠️  No Databricks token found (set DATABRICKS_TOKEN env var or databricks secret); skipping job trigger")
                        else:
                            # Get workspace URL
                            workspace_url = os.environ.get("DATABRICKS_WORKSPACE_URL", "https://adb-249008710733422.2.azuredatabricks.net")
                            if not workspace_url.startswith("http"):
                                workspace_url = f"https://{workspace_url}"
                            
                            # Build payload based on which job we're triggering
                            api_url = f"{workspace_url}/api/2.1/jobs/run-now"
                            
                            if trigger_job_id == args.phase6_job_id:
                                # Trigger Phase 6 (which will trigger Phase 5)
                                p6_params = [
                                    "--date", processing_date,
                                    "--selected-features", args.selected_features_path,
                                    "--delta-path", args.out_dir,
                                    "--trigger-phase5",
                                    "--phase5-job-id", str(args.phase5_job_id)
                                ]
                                # Forward model path if provided (CRITICAL for using new models)
                                if args.model_path:
                                    p6_params.extend(["--model-path", args.model_path])

                                # Forward skip-training flag if set (for initial deployment)
                                if args.skip_training:
                                    p6_params.append("--skip-training")

                                payload = {
                                    "job_id": trigger_job_id,
                                    "python_params": p6_params
                                }
                            else:
                                # Trigger Phase 5 directly (Cloudflare KV version)
                                payload = {
                                    "job_id": trigger_job_id,
                                    "python_params": [
                                        "--delta-path", args.out_dir,
                                        "--date", processing_date,
                                        "--selected-features", args.selected_features_path,
                                        # NOTE: No --model-path - Phase 5 will auto-detect latest model
                                        # NOTE: No --kv-config - Phase 5 will auto-detect from default path (dbfs:/mnt/bandit/config/cloudflare_kv_config.json)
                                        "--table-name", "spaceplay.user_multipliers",
                                        "--decisions-table", "spaceplay.bandit_decisions",
                                        "--inference"
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
                            print(f"   ✅ Triggered {job_name} (Job ID: {trigger_job_id}, Run ID: {run_id})")
                            print(f"   🔗 View run: {workspace_url}/#job/{trigger_job_id}/run/{run_id}")
                            
                            if trigger_job_id == args.phase6_job_id:
                                print(f"   📊 Pipeline: Phase 0 → Phase 6 (Online Learning) → Phase 5 (Batch Inference)")
                            
                    except Exception as e:
                        print(f"   ⚠️  Failed to trigger {job_name}: {e}")
                        import traceback
                        traceback.print_exc()
                        
                        # Fallback: Try to trigger Phase 5 directly if Phase 6 failed
                        if trigger_job_id == args.phase6_job_id and args.phase5_job_id:
                            print(f"   🔄 Attempting fallback to Phase 5 (Cloudflare KV) direct trigger...")
                            try:
                                fallback_payload = {
                                    "job_id": args.phase5_job_id,
                                    "python_params": [
                                        "--delta-path", args.out_dir,
                                        "--date", processing_date,
                                        "--selected-features", args.selected_features_path,
                                        # NOTE: No --model-path - Phase 5 will auto-detect latest model
                                        # NOTE: No --kv-config - Phase 5 will auto-detect from default path (dbfs:/mnt/bandit/config/cloudflare_kv_config.json)
                                        "--table-name", "spaceplay.user_multipliers",
                                        "--decisions-table", "spaceplay.bandit_decisions",
                                        "--inference"
                                    ]
                                }
                                
                                fallback_response = requests.post(
                                    api_url,
                                    headers={"Authorization": f"Bearer {databricks_token}"},
                                    json=fallback_payload,
                                    timeout=10
                                )
                                fallback_response.raise_for_status()
                                fallback_run_id = fallback_response.json().get("run_id")
                                print(f"   ✅ Fallback successful: Triggered Phase 5 (Job ID: {args.phase5_job_id}, Run ID: {fallback_run_id})")
                            except Exception as fallback_error:
                                print(f"   ❌ Fallback also failed: {fallback_error}")
        
        print("\n" + "=" * 80)
        print("PHASE 0 COMPLETE")
        print("=" * 80)
        print(f"   Output directory: {args.out_dir}")
        print(f"   Partition column: {args.partition_column}")
        print(f"   Output format: {args.output_format}")
        if args.inference and processing_date:
            print(f"   Processing date: {processing_date}")
            if args.phase5_date_output:
                print(f"   Date saved to: {args.phase5_date_output}")
        print("✅ Daily bandit pipeline (pure Spark) complete!")

    except Exception as exc:
        print(f"❌ Pipeline failed: {exc}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Don't stop SparkSession in Databricks - it's managed by the platform
        # In Databricks Jobs, SparkSession lifecycle is managed automatically
        try:
            # Databricks Python jobs don't automatically expose `dbutils`, so rely on env vars
            in_databricks = bool(
                os.environ.get("DATABRICKS_RUNTIME_VERSION")
                or os.environ.get("DATABRICKS_HOST")
                or os.environ.get("DB_CLUSTER_ID")
            )
            if in_databricks:
                print("ℹ️  Spark session cleanup handled by Databricks runtime (manual stop disabled)")
            else:
                print("ℹ️  Spark session left running (manual stop disabled per policy)")
        except Exception as e:
            # If anything fails, just log it (don't fail the job)
            print(f"ℹ️  Spark session cleanup: {e}")


if __name__ == "__main__":
    main()
