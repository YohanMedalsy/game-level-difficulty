# Game Level Difficulty

Contextual bandit pipeline and off-policy evaluation tooling for adaptive game difficulty.

## Structure
- `bandit/` - Training, evaluation, and serving pipeline for contextual bandits
- `ope/` - Off-policy evaluation utilities and experiments

## Config setup
Copy the example configs in `bandit/config/` and fill in the real values:
- `pipeline_params.example.json`
- `redis_config.example.json`
- `cloudflare_kv_config.example.json`

## Quickstart
```bash
pip install -r bandit/requirements.txt
```

See `bandit/README.md` for detailed pipeline steps and `ope/requirements_obp.txt` for OPE dependencies.
