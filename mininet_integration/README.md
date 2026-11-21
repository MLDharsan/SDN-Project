# Mininet validation backend (OS3E)

```
same traffic trace -> Mininet/OVS/os-ken -> same trained RDQN -> emulated metrics
```

This folder adds a Mininet emulation backend without changing the simulator.
No file in `environments/`, `experiments/`, `utils/`, `forecasting/`, `models/`
or `results/` is modified. All output goes to `mininet_results/`. Writing
anywhere else is refused in code (`_paths.assert_safe_output_dir`). That folder
is also outside the paths that `utils/archive_and_clean_results.py` cleans.

## How it works

| Piece | File | Role |
|---|---|---|
| Topology | `topology_zoo_mininet.py` | Builds the OS3E graph from the simulator's own `TopologyLoader`, using the same switch order and geographic link delays. One host per switch on port 1. Precomputes shortest-path routes by delay. |
| Controllers | `controller_manager.py`, `osken_apps/proactive_routing_app.py` | Controllers 1–5 are separate `osken-manager` processes on ports 6653–6657. Controller 0 (the master) is the Python orchestrator. Every controller installs the same routes, logs `switch_connected` and `flows_ready` (confirmed by a barrier reply), and drops table misses, so nothing is flooded on the cyclic graph. |
| Traffic | `traffic_replay.py` | A trace value `x` becomes UDP iperf3 at `x * max_rate_mbps` from host `h_i` to a fixed partner. Partners are chosen within the same connected component. RTT is probed with ping while traffic runs. |
| Telemetry | `telemetry.py` | Bytes entering each switch from its host (header overhead removed) are divided by `max_rate_mbps`, giving loads on the same 0–1 scale as the trace. Controller CPU and memory are reported but never given to the model. Also reads the real switch-to-controller mapping from OVSDB. |
| Policy | `policy_adapter.py` | `ShadowState` is an unchanged `ThresholdBasedProactiveSDN` fed with measured loads. It produces the 46-value (current) or 86-value (predictive) state, the valid-action mask, the migration targets, the park/evoke safety checks and the reward. |
| Execution | `action_executor.py` | Applies the shadow's decision to the network in this order: start the evoked controller, then `ovs-vsctl set-controller` for each migrated switch (waiting for `flows_ready`), then stop the parked controller. A controller is never stopped while it still owns switches. |
| Baselines | `mininet_baselines.py` | rdqn, threshold, random_valid and noop, using the selection functions imported from `evaluate_fair_baselines.py`. Also runs a simulator reference over the same trace window. |

**Validation:** in `--dry-run` with zero noise, the measured loads are the
trace rows exactly. The pipeline then reproduces the simulator exactly: actions
agree 100% and all metrics are identical, for all four policies, in both
current and predictive modes. This is covered by the tests.

## One-time setup

```bash
sudo apt install iperf3          # the only missing tool
venv/bin/python -m mininet_integration.check_environment
```

Nothing needs installing into the venv. The system Mininet package is exposed
through `mininet_integration/_vendor/mininet`, a symlink created automatically.

## Run order (matches the plan)

```bash
cd ~/sdn-drl-project
# no root
venv/bin/python -m unittest mininet_integration.tests.test_mininet_integration -v
venv/bin/python -m mininet_integration.run_mininet_experiment --dry-run --steps 50

# root: smoke tests
sudo venv/bin/python -m mininet_integration.smoke_test --topo single
sudo venv/bin/python -m mininet_integration.smoke_test --topo linear3 --traffic --migrate
sudo venv/bin/python -m mininet_integration.smoke_test --topo os3e --traffic --migrate

# root: experiment (start small)
sudo venv/bin/python -m mininet_integration.run_mininet_experiment --policies noop threshold --steps 20
sudo venv/bin/python -m mininet_integration.run_mininet_experiment --steps 120
sudo venv/bin/python -m mininet_integration.run_mininet_experiment --mode predictive --steps 120

# after a crash
sudo venv/bin/python -m mininet_integration.cleanup
```

Useful flags: `--profile real_case_study`, `--start-index`, `--interval`,
`--run-name`, `--skip-simulator-reference`. Press Ctrl+C to stop early; the
partial results are kept.

Each step takes about `interval_s` seconds (5 s by default), plus the time to
carry out any migrations. 120 steps with 4 policies is roughly 45 minutes.

## Output (`mininet_results/<run_id>/`)

- `comparison.csv`: per policy, emulated and simulator rows, using the
  fair-baseline metric names.
- `<policy>/summary.json`: simulator-style metrics plus emulation metrics:
  calibration error (measured vs trace load), packet loss, jitter, RTT,
  throughput, migration handover and flow-install delay, controller start/stop
  time, failed or inconsistent network operations, controller CPU/RSS,
  forecast MAE against measured loads, and action agreement with the simulator.
- `<policy>/emulated_steps.jsonl`: every step's loads, action, OVS operations
  and traffic stats.
- `run_info.json`: git commit and SHA-256 hashes of the model, GRU and trace
  files.

## What can and cannot be claimed

- **Measured:** throughput, loss, jitter, host RTT, controller CPU/memory,
  flow-installation delay, handover delay, migration success.
- **Simulated (labelled in outputs):** energy (the controller-state energy model)
  and controller-switch latency (graph-based). All controllers run on one machine,
  so control-channel latency is not emulated.
- **Fail-mode is `secure` and flows are proactive:** a handover does not
  interrupt forwarding. The measured handover cost is a control-plane cost.

## Note: OS3E topology file is disconnected

`data/topologies/Os3e.gml` has 36 edges forming two components: 28 nodes, plus
6 nodes (Athens, Belgrade, Bucharest, Istanbul, Kiev, Moscow) with no link to
the rest. Mininet reproduces this faithfully: routes and traffic stay inside
each component. In the simulator, `TopologyLoader` gives those 6 switches a
shortest-path delay of 0 ms to every controller, because unreachable pairs
default to 0.0. That lowers the reported average latency. This module does not
change that, but you should be aware of it.
