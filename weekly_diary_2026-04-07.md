# Weekly Diary Notes - 2026-04-07

## Objective

Implement the highest-priority paper-alignment items before the next training round:

1. Replace star/random topology distances with real topology delays.
2. Replace the topology-loader stub with real `.gml` parsing.
3. Add paper-style metrics to training/evaluation.
4. Fix evaluation consistency so custom Rainbow `.pth` checkpoints are evaluated directly.
5. Continue resolving the train/test mismatch with valid-action masking diagnostics.

## Implemented Changes

### 1. Real topology parsing

- Replaced the stub in [utils/topology_loader.py](/home/dharshan/sdn-drl-project/utils/topology_loader.py).
- Added support for:
  - loading Topology Zoo `.gml` files from `data/topologies/`
  - handling duplicate-edge GML files by parsing as multigraphs when needed
  - extracting node coordinates from GML attributes
  - fallback city coordinates for `Os3e.gml`
  - computing geographic edge delays using haversine distance
  - computing all-pairs shortest-path delay matrices
  - selecting controller host nodes from the real graph

### 2. Environment now uses real graph delays

- Updated [environments/threshold_proactive_sdn_env.py](/home/dharshan/sdn-drl-project/environments/threshold_proactive_sdn_env.py).
- Removed the old dependency on:
  - `nx.star_graph(...)`
  - random switch-to-controller distances
- The environment now:
  - loads the real graph through `TopologyLoader`
  - builds `distance_matrix` from real shortest-path delays
  - initializes switch-controller assignments using nearest active controller

### 3. Added paper-style metrics

- Added/propagated:
  - `cs_avg_latency`
  - `cs_worst_latency`
  - `load_balance_index`
  - `training_time_seconds`
- These are now surfaced through:
  - environment info dict
  - masked test harness
  - training metadata
  - evaluation script

Note:
- `load_balance_index` is currently an inferred proxy based on `max total assigned load / mean total assigned load`.
- This is much more stable than the earlier attempts and closer to controller-placement reporting scale.
- It should still be treated as provisional until the exact formula from the paper is confirmed.

### 4. Evaluation consistency fix

- Rewrote [experiments/evaluate_threshold_model.py](/home/dharshan/sdn-drl-project/experiments/evaluate_threshold_model.py).
- Removed SB3 `DQN.load(...)` dependency for this evaluation path.
- The evaluator now:
  - loads custom Rainbow `.pth` checkpoints
  - supports optional real traffic input
  - supports `--mask-invalid-actions`
  - reports paper-style metrics
  - saves JSON summaries under `results/evaluations/`

### 5. Continued train/test mismatch work

- Previous masking work remains active in:
  - [experiments/testing/test_trained_models_FIXED.py](/home/dharshan/sdn-drl-project/experiments/testing/test_trained_models_FIXED.py)
  - [experiments/train_rainbow_fixed_rewards.py](/home/dharshan/sdn-drl-project/experiments/train_rainbow_fixed_rewards.py)
- Added:
  - action decoding support from the environment
  - oscillation counting in the test harness
  - masked training option in the training script

## Verification Performed

### Static verification

- Ran:

```bash
python -m py_compile \
  utils/topology_loader.py \
  environments/threshold_proactive_sdn_env.py \
  experiments/train_rainbow_fixed_rewards.py \
  experiments/testing/test_trained_models_FIXED.py \
  experiments/evaluate_threshold_model.py
```

- Result: passed

### Topology-loader verification

- Verified all five topologies load successfully:
  - Gridnet
  - BellCanada
  - OS3E
  - Interoute
  - Cogentco

### Environment verification

- Instantiated the real-topology environment and confirmed it emits:
  - real delay matrices
  - `cs_avg_latency`
  - `cs_worst_latency`
  - `load_balance_index`

### Evaluation-script smoke test

- Ran:

```bash
cd experiments
python evaluate_threshold_model.py \
  --topology gridnet \
  --episodes 1 \
  --steps-per-episode 50 \
  --mask-invalid-actions
```

- Result:
  - script completed successfully
  - custom Rainbow checkpoint loaded
  - real metrics reported and JSON saved

## Important Notes / Risks

### 1. Paper reference values are still marked unverified

- `PAPER_RESULTS` in the evaluation script are currently retained as manual reference entries.
- They are now explicitly marked:
  - `UNVERIFIED_MANUAL_ENTRY`
- Before final thesis claims, these values should be checked against the actual paper tables.

### 2. Load-balance formula still needs exact paper confirmation

- The current implementation is a practical proxy, not a guaranteed exact reproduction of the paper formula.
- This is good enough to continue engineering work and retraining, but not good enough yet for final publication-grade comparison claims.

### 3. NumPy / Torch warning still appears in this environment

- The environment still prints:
  - NumPy 1.x vs NumPy 2.x compiled-module warning
- Current scripts still run, but this should be cleaned up later for stability.

## Retraining Plan

Next step is to retrain the weak topologies with masked action selection:

```bash
cd /home/dharshan/sdn-drl-project
python experiments/train_rainbow_fixed_rewards.py --topology bellcanada --timesteps 100000 --mask-invalid-actions
python experiments/train_rainbow_fixed_rewards.py --topology os3e --timesteps 80000 --mask-invalid-actions
python experiments/train_rainbow_fixed_rewards.py --topology cogentco --timesteps 200000 --mask-invalid-actions
```

Then re-evaluate with:

```bash
cd /home/dharshan/sdn-drl-project/experiments/testing
python test_trained_models_FIXED.py --test-all --mask-invalid-actions
```

And paper-style comparison with:

```bash
cd /home/dharshan/sdn-drl-project/experiments
python evaluate_threshold_model.py --topology gridnet --mask-invalid-actions
```

## Suggested Next Engineering Steps

1. Retrain BellCanada, OS3E, and Cogentco with masking enabled.
2. Re-run masked test evaluation and compare:
   - proactive counts
   - oscillation rate
   - mask override rate
   - paper-style latency metrics
3. Confirm the exact paper load-balance formula.
4. Only after that, move on to:
   - ablations
   - seed study
   - controller-count sweeps
   - heuristic baselines
   - optimality-gap experiments
