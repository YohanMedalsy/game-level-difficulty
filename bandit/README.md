# Vowpal Wabbit Contextual Bandit Pipeline

**Production-ready Doubly Robust (DR) contextual bandit for SpacePlay difficulty optimization**

---

## 📋 Overview

This pipeline evolves the Off-Policy Evaluation (OPE) system to a full production Vowpal Wabbit contextual bandit that:

- **Learns** optimal difficulty policies from logged data
- **Optimizes** hyperparameters using Optuna (epsilon-greedy, bagging, SquareCB)
- **Validates** against OPE baseline (must beat 95% of uniform policy)
- **Deploys** with <10ms p99 inference latency
- **Adapts** via daily online learning with same-day reward feedback

---

## 🎯 Quick Start

### Prerequisites

```bash
# Install Vowpal Wabbit
brew install vowpal-wabbit  # macOS
# OR
sudo apt-get install vowpal-wabbit  # Linux

# Install Python dependencies
cd bandit
pip install -r requirements.txt
```

### Config setup

Copy the example configs in `bandit/config/` to matching `.json` files and fill in your values (the real files are gitignored):

```bash
cp bandit/config/cloudflare_kv_config.example.json bandit/config/cloudflare_kv_config.json
cp bandit/config/redis_config.example.json bandit/config/redis_config.json
cp bandit/config/pipeline_params.example.json bandit/config/pipeline_params.json
```

### Run Full Pipeline

```bash
# Phase 0: Data Generation from Snowflake (OPTIONAL if Delta file already exists)
# Generates Delta file with 520+ features from Snowflake raw events
python scripts/00_datagen_from_snowflake.py \
  --out-dir test_folder/daily_features_spark.delta \
  --start-date 2025-07-01 \
  --output-format delta

# Phase 1: Feature Selection (Delta/Parquet)
# Reads Delta dataset produced by Phase 0 (or existing Delta file)
python scripts/01_prepare_data_spark.py \
  --dataset-path test_folder/daily_features_spark.delta \
  --start-date 2025-10-01 \
  --end-date 2025-10-28 \
  --train-propensity

# Phase 2: Convert to VW CB_ADF Format
python scripts/02_convert_delta_to_vw_spark.py \
  --dataset-path test_folder/daily_features_spark.delta \
  --selected-features-json artifacts/selected_features_union_70.json \
  --output-dir data/processed/vw_streaming

# Phase 3: Train VW with Optuna (100 trials, ~2-4 hours)
python scripts/03_train_vw_optuna.py

# Phase 4: Comprehensive Validation
python scripts/04_validate_vw.py
```

**Or use the convenience script:**

```bash
./run_all.sh
```

---

## 🏗️ Architecture

### Bandit Structure

| Component | Description |
|-----------|-------------|
| **Arms** | 5 difficulty levels: Easierer (-0.12), Easier (-0.06), Same (0.00), Harder (+0.06), Harderer (+0.12) |
| **Context** | 50 features (selected from 520 via RF) - user metrics, session stats, economy, EWMA |
| **Reward** | Coins spent next day (proxy for engagement + monetization) |
| **Policy** | VW Doubly Robust CB with epsilon-greedy/bagging/SquareCB exploration |

### VW CB_ADF Format

```
shared |u user_lifetime:45 |s session_count:12 |e coins:1500
0:-500:0.18 |a arm:0 delta:-0.12 |c mult:0.88 feasible:1
1:-350:0.22 |a arm:1 delta:-0.06 |c mult:0.88 feasible:1
2:-280:0.35 |a arm:2 delta:0.00 |c mult:0.88 feasible:1
3:-420:0.15 |a arm:3 delta:0.06 |c mult:0.88 feasible:1
4:-610:0.10 |a arm:4 delta:0.12 |c mult:0.88 feasible:1
```

Format: `action:cost:probability |namespace feature:value ...`

---

## 📂 Directory Structure

```
bandit/
├── config/                     # YAML configurations
│   ├── vw_config_best.yaml     # Best hyperparams from Optuna
│   ├── data_config.yaml
│   └── optuna_config.yaml
│
├── data/
│   ├── raw/                    # Symlinks to processed CSVs
│   │   ├── daily_features_claude.csv
│   │   └── daily_features_ewma_claude.csv
│   └── processed/              # Pipeline outputs
│       ├── train_df.parquet    # 80% split
│       ├── valid_df.parquet    # 10% split
│       ├── test_df.parquet     # 10% split
│       ├── train.vw            # VW format
│       ├── valid.vw
│       └── test.vw
│
├── models/
│   ├── vw_bandit_dr_best.vw    # Trained VW model
│   ├── propensity_model.pkl    # Logistic regression
│   └── feature_scaler.pkl      # StandardScaler
│
├── artifacts/
│   ├── selected_features_50.json       # Feature names
│   ├── optuna_study.db                 # Study database
│   ├── optuna_optimization_history.html
│   ├── validation_report.md
│   └── propensity_calibration.png
│
├── logs/
│   ├── training/               # VW training logs
│   ├── inference/              # API logs (Phase 5)
│   └── online_learning/        # Daily update logs (Phase 6)
│
├── src/
│   ├── constants.py            # Bandit structure
│   ├── config.py               # YAML loaders
│   ├── data/                   # Data pipeline modules
│   ├── validation/             # OPE estimators
│   │   └── ope_estimators.py
│   └── ...
│
├── scripts/
│   ├── 01_prepare_data.py      # Phase 1
│   ├── 02_convert_to_vw.py     # Phase 2
│   ├── 03_train_vw_optuna.py   # Phase 3
│   └── 04_validate_vw.py       # Phase 4
│
├── requirements.txt
└── README.md
```

---

## 🚀 Pipeline Phases

### Phase 0: Data Generation from Snowflake

**Script:** `scripts/00_datagen_from_snowflake.py`

**What it does:**

1. Connects to Snowflake and queries raw game events
2. Performs comprehensive feature engineering (520+ features):
   - Daily aggregations (user-day level)
   - Advanced gameplay features
   - EWMA features (exponential moving averages)
   - Lag features (previous day stats)
3. Computes `next_day_reward` (coins spent next day)
4. Assigns `difficulty_arm` labels (Easierer, Easier, Same, Harder, Harderer)
5. Writes Delta/Parquet file for Phase 1

**Outputs:**

- `daily_features_spark.delta` (or `.parquet`) - Input for Phase 1

**Runtime:** ~30-60 minutes (depends on date range)

**Note:** Skip this phase if you already have a Delta file from a previous run or from the `ope/` folder.

---

### Phase 1: Data Preparation (Feature Selection)

**Script:** `scripts/01_prepare_data_spark.py` (Spark-native, preferred for Delta)
 
> Note: A pandas-based variant `scripts/01_prepare_data.py` exists for legacy CSV flows. The Spark path avoids pandas and scales to large datasets.

**What it does:**

1. Loads `daily_features_claude.csv` + `daily_features_ewma_claude.csv`
2. Preprocessing: one-hot encoding → 520 features
3. Splits users 80/10/10 (train/valid/test), stratified by engagement
4. **RF feature selection**: 520 → **50 features** (arm-weighted, no leakage)
5. **Stability validation**: Jaccard similarity >0.8 across 5 seeds
6. Trains **propensity model (Spark ML)**: multinomial LogisticRegression saved to `models/propensity_spark`
7. Exports: selected features, propensity model, scaler, data splits

**Outputs:**

- `artifacts/selected_features_50.json`
- `models/propensity_model.pkl`
- `models/feature_scaler.pkl`
- `data/processed/{train,valid,test}_df.parquet`

**Runtime:** ~5-10 minutes

---

### Phase 2: VW Data Conversion

**Script (Spark streaming):** `scripts/02_convert_delta_to_vw_spark.py`

**What it does:**

1. Loads Delta/Parquet dataset (date-filtered), no pandas
2. Deterministic user-based split: **80/10/10** (train/valid/test)
3. Organizes features into **VW namespaces** (u=user, s=session, e=economy, etc.)
4. Computes **logged propensities** using the saved Spark ML PipelineModel (or falls back to uniform over feasible arms)
5. Converts to **VW CB_ADF format**:
   - Shared context with namespaced features
   - 5 action lines (cost=-reward for observed arm, propensities)
   - Action features (arm ID, delta, current multiplier, feasibility)
6. Writes sharded VW files per split under `data/processed/vw_streaming/{train,valid,test}`
   - Phase 3 auto-concatenates shards into single files `train.vw`, `valid.vw`, `test.vw`.

**Outputs:**

- `data/processed/train.vw`
- `data/processed/valid.vw`
- `data/processed/test.vw`

**Runtime:** ~10-20 minutes (depends on dataset size)

---

## 📐 Propensity Training Modes (Spark ML)

- Default (Production): Train on all available rows
  - Pros: More samples → stronger behavior model → lower variance DR
  - Acceptable leakage for production since behavior policy is context-only
  - Command: `scripts/01_prepare_data_spark.py --train-propensity`

- Validation mode: Train on the 80% train split only
  - Pros: Strict separation from validation/test (no leakage)
  - Command: add `--use-train-split-for-propensity`

Label mapping verification
- The converter verifies the saved StringIndexer labels against the canonical `ARM_ORDER`.
- If the sets differ, it prints a warning and proceeds; the chosen action’s probability is extracted by the indexed label, so training remains correct.

---

### Phase 3: VW Training with Optuna

**Script:** `scripts/03_train_vw_optuna.py`

**What it does:**

1. **Multi-objective optimization** with Optuna (minimize loss, maximize DR)
2. **Hyperparameter search space:**
   - CB algorithm: `dr` (fixed - Doubly Robust)
   - Exploration: epsilon-greedy vs epsilon+bagging vs SquareCB
   - Learning rate: [1e-4, 1e-1] (log-uniform)
   - L2 regularization: [1e-6, 1e-2] (log-uniform)
   - Power_t (decay): [0.0, 0.5]
   - Feature interactions: None, ua, us, uas, all
   - Quadratic: True/False
   - Bagging: [3, 10] (for epsilon_bag)
3. **100 trials** with TPE sampler + median pruner
4. Selects **best trial** from Pareto front (prioritizes DR)
5. Trains **final model** with best hyperparameters
6. Generates **interactive visualizations** (Plotly)

**Outputs:**

- `models/vw_bandit_dr_best.vw`
- `config/vw_config_best.yaml`
- `artifacts/optuna_study.db`
- `artifacts/optuna_optimization_history.html`
- `artifacts/optuna_pareto_front.html`

**Runtime:** ~2-4 hours (100 trials, sequential)

---

### Phase 4: Comprehensive Validation

**Script:** `scripts/04_validate_vw.py`

**What it does:**

1. **OPE Comparison**:
   - Computes DR estimate on test set
   - Compares with OPE uniform baseline: **1234.86 ± 1006.60 coins/day**
   - **Pass criteria**: VW DR ≥ 95% of baseline (1173.12 coins/day)

2. **Propensity Calibration**:
   - Expected Calibration Error (ECE) per arm
   - Calibration curves (predicted vs empirical probabilities)
   - **Pass criteria**: Mean ECE < 0.05

3. **Inference Performance**:
   - Benchmarks 1000 VW predictions
   - Measures p50, p95, p99 latency
   - **Pass criteria**: p99 < 10ms

4. **Generates validation report** (Markdown)

**Outputs:**

- `artifacts/validation_report.md`
- `artifacts/vw_vs_ope_comparison.csv`
- `artifacts/propensity_calibration.png`

**Runtime:** ~5-10 minutes

---

## 📊 Key Results

### OPE Baseline (Uniform Policy)

From `difficulty_ope_daily_correct.py`:

| Metric | Value |
|--------|-------|
| **DR Estimate** | **1234.86 ± 1006.60** coins/day |
| **Significance** | p = 0.0202 (statistically significant vs 0) |
| **Dataset** | 2.9M observations, 33,501 users |

### VW Model Performance

Expected results after Optuna optimization:

| Metric | Target | Expected |
|--------|--------|----------|
| **Validation DR** | ≥ 1173.12 coins/day | 1300-1500 (+5-20% lift) |
| **Propensity ECE** | < 0.05 | 0.02-0.04 |
| **Inference p99** | < 10ms | 3-7ms |

---

## 🛠️ Configuration

### Data Configuration

**File:** `config/data_config.yaml`

```yaml
base_csv_path: data/raw/daily_features_claude.csv
ewma_csv_path: data/raw/daily_features_ewma_claude.csv

train_ratio: 0.8
valid_ratio: 0.1
test_ratio: 0.1

stratify_by: engagement_quantile
n_stratify_bins: 5

feature_config:
  selection_method: rf
  n_features: 50
  sample_frac: 0.8
  random_seed: 42
  stability_check: true
  stability_n_seeds: 5
  stability_jaccard_threshold: 0.8
```

### VW Configuration (Best Params)

**File:** `config/vw_config_best.yaml` (auto-generated by Optuna)

```yaml
cb_type: dr
exploration_type: epsilon_bag
epsilon: 0.08
bag: 5
learning_rate: 0.015
power_t: 0.0
l2: 0.0001
interactions: ua
quadratic: true
```

### Optuna Configuration

**File:** `config/optuna_config.yaml`

```yaml
study_name: vw_dr_bandit
n_trials: 100
n_jobs: 1
directions: [minimize, minimize]  # [val_loss, -val_dr]
pruner: median
n_startup_trials: 10

search_spaces:
  cb_type: [dr]
  exploration_type: [epsilon, epsilon_bag, squarecb]
  epsilon: [0.01, 0.3]
  bag: [3, 10]
  learning_rate: [0.0001, 0.1]
  power_t: [0.0, 0.5]
  l2: [0.000001, 0.01]
  interactions: [null, ua, us, uas]
  quadratic: [false, true]
```

---

## 🔬 Technical Details

### Feature Selection

**Method:** Random Forest arm-weighted importance

- Trains RF to predict **arm** (not reward) → avoids target leakage
- Samples 80% of training data for speed
- Selects top 50 features by importance
- Validates stability across 5 random seeds (Jaccard >0.8)

**Why 50 features?**

- Matches OPE pipeline (consistency)
- Reduces overfitting
- Faster VW training + inference
- Stable across different data samples

### VW CB Algorithms Compared

| Algorithm | Description | Use Case |
|-----------|-------------|----------|
| **dr** (Doubly Robust) ✅ | IPS + learned baseline | **Best for logged data with propensities** |
| ips | Pure importance weighting | High variance, simple |
| mtr | DR with automatic baseline | Black-box, less control |
| dm | Direct method (ignores propensities) | Biased, not recommended |

**We use DR because:**

1. Matches OPE methodology exactly
2. Lower variance than IPS
3. Unbiased (theoretically sound)
4. Leverages propensities optimally

### Exploration Strategies

| Strategy | Parameters | Pros | Cons |
|----------|------------|------|------|
| **Epsilon-greedy** | `--epsilon 0.1` | Simple, interpretable | Fixed exploration |
| **Epsilon + Bagging** ✅ | `--epsilon 0.05 --bag 5` | **Ensemble robustness, better uncertainty** | Slower training |
| **SquareCB** | `--squarecb --gamma_scale 10` | Variance-aware (UCB-like) | Newer, less tested |

**Optuna tests all 3** and selects the best for your data.

### Reward Timing (No Delay!)

```
End of Day T:
  1. Observe Day T reward (coins spent)
  2. Learn from Day T-1 decision (update VW model)
  3. Extract Day T features
  4. Decide Day T+1 difficulty
```

**Key insight:** Same-day feedback enables daily online learning.

---

## 🧪 Validation Criteria

### 1. OPE Comparison

- ✅ **PASS**: VW DR ≥ 1173.12 coins/day (95% of uniform baseline)
- ❌ **FAIL**: VW DR < 1173.12

### 2. Propensity Calibration

- ✅ **PASS**: Mean ECE < 0.05
- ⚠️  **WARNING**: 0.05 ≤ Mean ECE < 0.10
- ❌ **FAIL**: Mean ECE ≥ 0.10

### 3. Inference Performance

- ✅ **PASS**: p99 latency < 10ms
- ⚠️  **WARNING**: 10ms ≤ p99 < 50ms
- ❌ **FAIL**: p99 ≥ 50ms

---

## 📈 Next Steps (Phases 5-7)

### Phase 5: Production Inference Service

- FastAPI endpoint for real-time predictions
- Kubernetes deployment (3 replicas, auto-scaling)
- <10ms p99 latency, <50% CPU
- Graceful degradation (fallback to uniform)

### Phase 6: Online Learning

- **Daily cycle:**
  - 6am: Aggregate previous day's (context, action, reward)
  - 7am: Incremental VW update (`--initial_regressor`)
  - 8am: Deploy updated model (atomic swap)
- Monitor: daily DR, propensity drift, arm distribution

### Phase 7: A/B Testing

- **Gradual rollout:** 10% → 25% → 50% → 100%
- **DR-based lift estimation** (user-stratified bootstrap)
- **Sequential testing** (O'Brien-Fleming α-spending)
- **Automated rollback** if lift < -5%

---

## 🐛 Troubleshooting

### VW Command Not Found

```bash
# macOS
brew install vowpal-wabbit

# Ubuntu/Debian
sudo apt-get install vowpal-wabbit

# Or build from source
git clone https://github.com/VowpalWabbit/vowpal_wabbit.git
cd vowpal_wabbit
make
sudo make install
```

### Propensity Model Warning: "Some Arms Missing"

**Cause:** Test set has arms not seen during training.

**Fix:** This is expected if certain arms are rare. Propensity model assigns small probability (1e-6) to unseen arms.

### Optuna Trial Failures

**Cause:** VW command syntax errors or incompatible hyperparameters.

**Fix:** Check `logs/training/trial_*.log` for error messages. Common issues:

- `--squarecb` requires recent VW version (≥9.0)
- `--interactions` syntax: use `ua` not `u a`

### Low Effective Sample Size (ESS < 50%)

**Cause:** High variance in importance weights (propensity collapse).

**Fix:**

1. Check propensity calibration (ECE should be <0.05)
2. Increase propensity clipping (default: 10.0)
3. Use DR instead of IPS (already default)

---

## 📚 References

### Papers

1. **Doubly Robust Off-Policy Evaluation**
   Dudík, M., Langford, J., & Li, L. (2011)

2. **Contextual Bandit Benchmarks**
   Bietti, A., Agarwal, A., & Langford, J. (2021)

3. **Vowpal Wabbit Technical Report**
   Langford, J., Li, L., & Strehl, A. (2007)

### Documentation

- [Vowpal Wabbit Wiki](https://github.com/VowpalWabbit/vowpal_wabbit/wiki)
- [VW Contextual Bandit Tutorial](https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Contextual-Bandit-algorithms)
- [Optuna Documentation](https://optuna.readthedocs.io/)

### Related Code

- `difficulty_ope_daily_correct.py` - OPE baseline implementation
- `bandit_datagen_daily_claude.py` - Data generation pipeline

---

## 👥 Contributing

This is an internal research project. For questions or improvements:

1. Review `artifacts/validation_report.md` for current performance
2. Check `logs/training/*.log` for training details
3. Consult `config/*.yaml` for hyperparameters
4. Open an issue with reproduction steps

---

## 📄 License

Internal use only. Do not distribute without permission.

---

## ✅ Checklist: Production Readiness

- [x] Data generation from Snowflake (Phase 0)
- [x] Feature selection (Phase 1)
- [x] VW conversion (Phase 2)
- [x] Hyperparameter optimization (Phase 3)
- [x] Comprehensive validation (Phase 4)
- [ ] Inference API (Phase 5)
- [ ] Online learning (Phase 6)
- [ ] A/B testing framework (Phase 7)
- [ ] Monitoring dashboards
- [ ] Incident response runbook

**Current Status:** ✅ Ready for training and validation (Phases 0-4)

---

**Last Updated:** 2025-10-19
**Version:** 1.0.0
**Maintainer:** Data Science Team
