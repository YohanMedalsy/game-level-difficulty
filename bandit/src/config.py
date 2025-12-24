"""
Configuration Management for VW Bandit Pipeline

Loads and validates YAML configuration files for data prep, training, and deployment.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from dataclasses import dataclass, field


# Get bandit root directory (parent of src/)
BANDIT_ROOT = Path(__file__).parent.parent.absolute()
CONFIG_DIR = BANDIT_ROOT / "config"
DATA_DIR = BANDIT_ROOT / "data"
MODELS_DIR = BANDIT_ROOT / "models"
ARTIFACTS_DIR = BANDIT_ROOT / "artifacts"
LOGS_DIR = BANDIT_ROOT / "logs"


@dataclass
class FeatureConfig:
    """Feature selection configuration."""
    selection_method: str = "rf"  # 'rf' or 'lasso'
    n_features: int = 50
    sample_frac: float = 0.8
    random_seed: int = 42
    stability_check: bool = True
    stability_n_seeds: int = 5
    stability_jaccard_threshold: float = 0.8


@dataclass
class DataConfig:
    """Data preparation configuration."""
    base_csv_path: str = str(DATA_DIR / "raw" / "daily_features_claude.csv")
    ewma_csv_path: str = str(DATA_DIR / "raw" / "daily_features_ewma_claude.csv")

    # Train/valid/test splits
    train_ratio: float = 0.8
    valid_ratio: float = 0.1
    test_ratio: float = 0.1

    # Stratification
    stratify_by: str = "engagement_quantile"  # or "user_lifetime_quantile"
    n_stratify_bins: int = 5

    # Random seed for reproducibility
    random_seed: int = 42

    # Output paths
    output_dir: str = str(DATA_DIR / "processed")

    # Feature configuration
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)


@dataclass
class VWConfig:
    """Vowpal Wabbit training configuration."""

    # CB algorithm
    cb_type: str = "dr"  # 'ips', 'dr', 'mtr', 'dm'

    # Exploration strategy (will be optimized by Optuna)
    exploration_type: str = "epsilon_bag"  # 'epsilon', 'epsilon_bag', 'squarecb'
    epsilon: float = 0.08
    bag: int = 5
    gamma_scale: float = 10.0  # For squarecb

    # Core hyperparameters (will be optimized by Optuna)
    learning_rate: float = 0.015
    power_t: float = 0.0  # Learning rate decay
    l2: float = 0.0001

    # Feature interactions
    interactions: Optional[str] = "ua"  # None, 'ua', 'us', 'uas'
    quadratic: bool = True  # All pairwise interactions
    cubic: bool = False     # All 3-way interactions

    # Training settings
    passes: int = 3
    cache_file: bool = True
    holdout_off: bool = True

    # Model paths
    model_output_path: str = str(MODELS_DIR / "vw_bandit_dr_best.vw")

    # Online learning (Phase 6)
    online_learning_rate: float = 0.001  # Lower LR for online updates


@dataclass
class OptunaConfig:
    """Optuna hyperparameter optimization configuration."""

    # Study settings
    study_name: str = "vw_dr_bandit"
    storage: str = f"sqlite:///{ARTIFACTS_DIR / 'optuna_study.db'}"
    n_trials: int = 100
    n_jobs: int = -1  # Use all cores

    # Multi-objective optimization
    directions: list = field(default_factory=lambda: ["minimize", "minimize"])  # [val_loss, -val_dr]

    # Pruning
    pruner: str = "median"  # 'median', 'hyperband', 'none'
    n_startup_trials: int = 10

    # Hyperparameter search spaces
    search_spaces: Dict[str, Any] = field(default_factory=lambda: {
        'cb_type': ['dr'],  # Fixed to DR
        'exploration_type': ['epsilon', 'epsilon_bag', 'squarecb'],
        'epsilon': [0.01, 0.3],
        'bag': [3, 10],
        'gamma_scale': [1.0, 100.0],
        'learning_rate': [1e-4, 1e-1],
        'power_t': [0.0, 0.5],
        'l2': [1e-6, 1e-2],
        'interactions': [None, 'ua', 'us', 'uas'],
        'quadratic': [False, True],
    })

    # Constraints
    max_inference_latency_ms: float = 10.0  # p99 latency threshold

    # Visualization
    create_plots: bool = True
    plot_output_dir: str = str(ARTIFACTS_DIR)


@dataclass
class ValidationConfig:
    """Validation configuration."""

    # OPE comparison thresholds (from constants.py)
    ope_uniform_dr_mean: float = 1234.86
    ope_uniform_dr_std: float = 1006.60

    # Validation thresholds
    vw_dr_min_ratio: float = 0.95  # VW must beat 95% of uniform
    vw_dr_max_ratio: float = 1.5   # Sanity check upper bound

    # Propensity calibration
    expected_calibration_error_threshold: float = 0.05
    n_calibration_bins: int = 10

    # Feature importance
    feature_importance_jaccard_threshold: float = 0.4

    # Inference performance
    latency_p99_threshold_ms: float = 10.0
    n_latency_benchmark_requests: int = 1000

    # Bootstrap for confidence intervals
    n_bootstrap_samples: int = 10000
    bootstrap_ci_alpha: float = 0.05


@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    vw: VWConfig = field(default_factory=VWConfig)
    optuna: OptunaConfig = field(default_factory=OptunaConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)


def load_yaml_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    return config_dict or {}


def create_default_configs():
    """
    Create default configuration YAML files if they don't exist.
    This is called on first setup to generate template configs.
    """
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Data config
    data_config_path = CONFIG_DIR / "data_config.yaml"
    if not data_config_path.exists():
        data_config = {
            'base_csv_path': str(DATA_DIR / "raw" / "daily_features_claude.csv"),
            'ewma_csv_path': str(DATA_DIR / "raw" / "daily_features_ewma_claude.csv"),
            'train_ratio': 0.8,
            'valid_ratio': 0.1,
            'test_ratio': 0.1,
            'stratify_by': 'engagement_quantile',
            'n_stratify_bins': 5,
            'random_seed': 42,
            'output_dir': str(DATA_DIR / "processed"),
            'feature_config': {
                'selection_method': 'rf',
                'n_features': 50,
                'sample_frac': 0.8,
                'random_seed': 42,
                'stability_check': True,
                'stability_n_seeds': 5,
                'stability_jaccard_threshold': 0.8,
            }
        }
        with open(data_config_path, 'w') as f:
            yaml.dump(data_config, f, default_flow_style=False, sort_keys=False)
        print(f"✅ Created: {data_config_path}")

    # VW config (will be updated by Optuna)
    vw_config_path = CONFIG_DIR / "vw_config.yaml"
    if not vw_config_path.exists():
        vw_config = {
            'cb_type': 'dr',
            'exploration_type': 'epsilon_bag',
            'epsilon': 0.08,
            'bag': 5,
            'gamma_scale': 10.0,
            'learning_rate': 0.015,
            'power_t': 0.0,
            'l2': 0.0001,
            'interactions': 'ua',
            'quadratic': True,
            'cubic': False,
            'passes': 3,
            'cache_file': True,
            'holdout_off': True,
            'model_output_path': str(MODELS_DIR / "vw_bandit_dr_best.vw"),
            'online_learning_rate': 0.001,
        }
        with open(vw_config_path, 'w') as f:
            yaml.dump(vw_config, f, default_flow_style=False, sort_keys=False)
        print(f"✅ Created: {vw_config_path}")

    # Optuna config
    optuna_config_path = CONFIG_DIR / "optuna_config.yaml"
    if not optuna_config_path.exists():
        optuna_config = {
            'study_name': 'vw_dr_bandit',
            'storage': f"sqlite:///{ARTIFACTS_DIR / 'optuna_study.db'}",
            'n_trials': 100,
            'n_jobs': -1,
            'directions': ['minimize', 'minimize'],
            'pruner': 'median',
            'n_startup_trials': 10,
            'search_spaces': {
                'cb_type': ['dr'],
                'exploration_type': ['epsilon', 'epsilon_bag', 'squarecb'],
                'epsilon': [0.01, 0.3],
                'bag': [3, 10],
                'gamma_scale': [1.0, 100.0],
                'learning_rate': [0.0001, 0.1],
                'power_t': [0.0, 0.5],
                'l2': [0.000001, 0.01],
                'interactions': [None, 'ua', 'us', 'uas'],
                'quadratic': [False, True],
            },
            'max_inference_latency_ms': 10.0,
            'create_plots': True,
            'plot_output_dir': str(ARTIFACTS_DIR),
        }
        with open(optuna_config_path, 'w') as f:
            yaml.dump(optuna_config, f, default_flow_style=False, sort_keys=False)
        print(f"✅ Created: {optuna_config_path}")

    # Validation config
    validation_config_path = CONFIG_DIR / "validation_config.yaml"
    if not validation_config_path.exists():
        validation_config = {
            'ope_uniform_dr_mean': 1234.86,
            'ope_uniform_dr_std': 1006.60,
            'vw_dr_min_ratio': 0.95,
            'vw_dr_max_ratio': 1.5,
            'expected_calibration_error_threshold': 0.05,
            'n_calibration_bins': 10,
            'feature_importance_jaccard_threshold': 0.4,
            'latency_p99_threshold_ms': 10.0,
            'n_latency_benchmark_requests': 1000,
            'n_bootstrap_samples': 10000,
            'bootstrap_ci_alpha': 0.05,
        }
        with open(validation_config_path, 'w') as f:
            yaml.dump(validation_config, f, default_flow_style=False, sort_keys=False)
        print(f"✅ Created: {validation_config_path}")


def load_config(config_name: str = "data_config") -> Config:
    """
    Load configuration from YAML file.

    Args:
        config_name: Name of config file (without .yaml extension)

    Returns:
        Config object
    """
    config_path = CONFIG_DIR / f"{config_name}.yaml"

    if not config_path.exists():
        print(f"⚠️  Config file not found: {config_path}")
        print(f"Creating default configurations...")
        create_default_configs()

    config_dict = load_yaml_config(config_path)

    # Map config name to dataclass
    if config_name == "data_config":
        return DataConfig(**config_dict)
    elif config_name == "vw_config":
        return VWConfig(**config_dict)
    elif config_name == "optuna_config":
        return OptunaConfig(**config_dict)
    elif config_name == "validation_config":
        return ValidationConfig(**config_dict)
    else:
        raise ValueError(f"Unknown config name: {config_name}")


def load_all_configs() -> Config:
    """Load all configuration files and combine into master Config object."""
    # Ensure default configs exist
    create_default_configs()

    # Load each config
    data_config = load_config("data_config")
    vw_config = load_config("vw_config")
    optuna_config = load_config("optuna_config")
    validation_config = load_config("validation_config")

    # Combine into master config
    config = Config(
        data=data_config,
        vw=vw_config,
        optuna=optuna_config,
        validation=validation_config
    )

    return config


if __name__ == "__main__":
    # Test configuration system
    print("VW Bandit Configuration System")
    print("=" * 80)

    # Create default configs
    print("\n1. Creating default configuration files...")
    create_default_configs()

    # Load master config
    print("\n2. Loading master configuration...")
    config = load_all_configs()

    print(f"\n3. Configuration summary:")
    print(f"   Data:")
    print(f"     - Training ratio: {config.data.train_ratio}")
    print(f"     - Features selected: {config.data.feature_config.n_features}")
    print(f"     - Random seed: {config.data.random_seed}")

    print(f"\n   VW:")
    print(f"     - CB algorithm: {config.vw.cb_type}")
    print(f"     - Exploration: {config.vw.exploration_type}")
    print(f"     - Learning rate: {config.vw.learning_rate}")
    print(f"     - L2 regularization: {config.vw.l2}")

    print(f"\n   Optuna:")
    print(f"     - Trials: {config.optuna.n_trials}")
    print(f"     - Study name: {config.optuna.study_name}")
    print(f"     - Jobs: {config.optuna.n_jobs}")

    print(f"\n   Validation:")
    print(f"     - OPE baseline: {config.validation.ope_uniform_dr_mean:.2f}")
    print(f"     - VW must beat: {config.validation.ope_uniform_dr_mean * config.validation.vw_dr_min_ratio:.2f}")
    print(f"     - ECE threshold: {config.validation.expected_calibration_error_threshold}")

    print("\n" + "=" * 80)
    print("✅ Configuration system validated successfully!")
