## Diagnostics Outputs

Artifacts are written to `artifacts/diagnostics/` after Phase 4 (`04_validate_vw.py`). Each file is safe to monitor or ingest into dashboards. The markdown validation report now includes a **Diagnostics Summary** section that surfaces the key metrics and alerts below for human review.

| File | Description | Alert Guidance |
| --- | --- | --- |
| `weights_summary.json` | Importance-weight stats (quantiles, clipping, ESS). | `ess_ratio < 0.5` → investigate; `clip_rates["100"] > 0.05` or `extreme_weights_count_100 > 0` → warning. |
| `weights_hist.png` | Log-scale histogram of importance weights. | Visual inspection for heavy tails. |
| `coverage_summary.json` | Arm counts, percentages, entropy/std, rare-arm list, user ID parsing metrics. | Counts `< min_arm_fail` (default 50) should trigger response; `user_id_parsing.success_rate < 0.95` indicates upstream data issues. |
| `validation_summary.json` | Aggregated metrics plus alert flags (low ESS, coverage breaches, extreme weights, user ID rate). | Consume in monitoring systems for automated alerting. |
| `validation_metrics.csv` | Flat key/value export for dashboards. | Suitable for Prometheus / Grafana ingestion. |
| `dr_bootstrap.json` | Bootstrap DR/IPS/SNIPS CIs using pandas propensity path (if available). | Provides uncertainty for uniform policy comparison. |
| `dr_bootstrap_vw_policy.json` | Bootstrap DR/IPS/SNIPS CIs using VW target policy (`--vw-target-policy-ci`). | Monitors statistical stability of production policy. |
| `vw_feature_importance.json` | Top VW hashed weights, namespace breakdown, overlaps with Spark selections. | Helps explainability and feature drift analysis. |
| `vw_readable.txt` | Raw output from `vw --invert_hash` used for importance parsing. | Useful for deep dives; large files can be rotated. |

### Thresholds

Thresholds are currently command-line parameters (`04_validate_vw.py`):

- `--clip-thresholds` (default `10,20,50,100`)
- `--min-arm-warn` (default `20`)
- `--min-arm-fail` (default `50`)
- `--bootstrap-samples` (default `1000`)

Future work: Promote these to `bandit/config/diagnostics.yaml` for centralized tuning.

### User ID Parsing

User IDs are written in the shared line comment (`# uid:<id>`) for each CB-ADF example. Diagnostic files include parsing success metrics. If the success rate drops, investigate upstream data generation to ensure user IDs are emitted.
