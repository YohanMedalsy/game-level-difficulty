#!/bin/bash
#
# Run All VW Bandit Pipeline Phases
#
# Usage: ./run_all.sh
#

set -e  # Exit on error

BANDIT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$BANDIT_ROOT"

echo "================================================================================"
echo "              VW CONTEXTUAL BANDIT PIPELINE - FULL EXECUTION"
echo "================================================================================"
echo ""
echo "This script will run all 4 phases sequentially:"
echo "  Phase 1: Data Preparation (~5-10 min)"
echo "  Phase 2: VW Conversion (~10-20 min)"
echo "  Phase 3: Optuna Training (~2-4 hours)"
echo "  Phase 4: Validation (~5-10 min)"
echo ""
echo "Total estimated time: ~3-5 hours"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Aborted."
    exit 1
fi

echo ""
echo "================================================================================"
echo "PHASE 1: DATA PREPARATION"
echo "================================================================================"

# Prefer Spark-native feature selection if dataset path provided
if [ -n "$BANDIT_DATASET_PATH" ]; then
  echo "Using Spark-native Phase 1 (feature selection) with dataset: $BANDIT_DATASET_PATH"
  python scripts/01_prepare_data_spark.py \
    --dataset-path "$BANDIT_DATASET_PATH" \
    --start-date "${BANDIT_START_DATE:-2025-10-01}" \
    ${BANDIT_END_DATE:+--end-date "$BANDIT_END_DATE"} \
    --train-propensity
else
  python scripts/01_prepare_data.py
fi
if [ $? -ne 0 ]; then
    echo "❌ Phase 1 failed!"
    exit 1
fi
echo "✅ Phase 1 complete"

echo ""
echo "================================================================================"
echo "PHASE 2: VW DATA CONVERSION"
echo "================================================================================"

# Prefer Spark streaming converter when dataset path is provided
if [ -n "$BANDIT_DATASET_PATH" ]; then
  OUT_DIR="data/processed/vw_streaming"
  echo "Using Spark streaming converter: $BANDIT_DATASET_PATH → $OUT_DIR"
  python scripts/02_convert_delta_to_vw_spark.py \
    --dataset-path "$BANDIT_DATASET_PATH" \
    --start-date "${BANDIT_START_DATE:-2025-10-01}" \
    ${BANDIT_END_DATE:+--end-date "$BANDIT_END_DATE"} \
    --output-dir "$OUT_DIR" \
    --selected-features-json artifacts/selected_features_union_70.json \
    ${PROPENSITY_MODEL:+--propensity-model "$PROPENSITY_MODEL"}
  # Optional: concatenate shards to single VW files for Phase 3
  # find "$OUT_DIR" -type f -name 'part-*' -exec cat {} + > data/processed/train.vw
  # For now, we keep sharded files; adjust Phase 3 if needed.
else
  python scripts/02_convert_to_vw.py
fi
if [ $? -ne 0 ]; then
    echo "❌ Phase 2 failed!"
    exit 1
fi
echo "✅ Phase 2 complete"

echo ""
echo "================================================================================"
echo "PHASE 3: VW TRAINING WITH OPTUNA"
echo "================================================================================"
echo "⏱️  This will take ~2-4 hours (100 trials)"
echo ""
python scripts/03_train_vw_optuna.py
if [ $? -ne 0 ]; then
    echo "❌ Phase 3 failed!"
    exit 1
fi
echo "✅ Phase 3 complete"

echo ""
echo "================================================================================"
echo "PHASE 4: COMPREHENSIVE VALIDATION"
echo "================================================================================"
python scripts/04_validate_vw.py --vw-target-policy-ci --bootstrap-samples 1000
if [ $? -ne 0 ]; then
    echo "❌ Phase 4 failed!"
    exit 1
fi
echo "✅ Phase 4 complete"

echo ""
echo "================================================================================"
echo "🎉 PIPELINE COMPLETE!"
echo "================================================================================"
echo ""
echo "📊 Results:"
echo "  • VW Model:         models/vw_bandit_dr_best.vw"
echo "  • Best Config:      config/vw_config_best.yaml"
echo "  • Validation Report: artifacts/validation_report.md"
echo "  • Optuna Study:     artifacts/optuna_study.db"
echo ""
echo "🔗 View Optuna results:"
echo "  optuna-dashboard artifacts/optuna_study.db"
echo ""
echo "📈 Next Steps:"
echo "  • Review artifacts/validation_report.md"
echo "  • Check if VW DR ≥ OPE baseline (1173.12 coins/day)"
echo "  • Proceed to Phase 5 (Inference API) if validation passes"
echo ""
echo "================================================================================"
