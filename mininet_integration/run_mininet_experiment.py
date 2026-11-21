#!/usr/bin/env python3
"""Run the OS3E Mininet validation experiment.

Live emulation (needs root, run from the project root):

    sudo venv/bin/python -m mininet_integration.run_mininet_experiment \
        --policies threshold noop --steps 20

Dry run (no root, no network; measured loads = trace rows). With zero noise
this must reproduce the simulator exactly and is used to validate the
pipeline:

    venv/bin/python -m mininet_integration.run_mininet_experiment --dry-run

Outputs go only to ``mininet_results/<run_id>/``:
    config_used.json, run_info.json, topology_spec.json,
    <policy>/emulated_steps.jsonl, <policy>/simulator_steps.jsonl,
    <policy>/summary.json, <policy>/controllers/*.jsonl|log,
    comparison.csv, comparison.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

from mininet_integration import _paths  # noqa: E402
from mininet_integration._paths import (  # noqa: E402
    PROJECT_ROOT,
    RESULTS_ROOT,
    assert_safe_output_dir,
    chown_to_invoking_user,
)
from mininet_integration.action_executor import (  # noqa: E402
    ActionExecutor,
    DryRunBackend,
    LiveMininetBackend,
    compute_diff,
)
from mininet_integration.config import load_config  # noqa: E402
from mininet_integration.mininet_baselines import (  # noqa: E402
    POLICIES,
    STEP_METRIC_KEYS,
    run_simulator_reference,
    summarize_decisions,
)
from mininet_integration.policy_adapter import (  # noqa: E402
    PolicySelector,
    ShadowState,
    make_simulator_env,
)
from mininet_integration.topology_zoo_mininet import build_topology_spec  # noqa: E402
from mininet_integration.traffic_replay import warmup_rows  # noqa: E402

ENERGY_NOTE = "Simulated controller-state energy model (not measured by Mininet)."
LATENCY_NOTE = (
    "cs_avg_latency / cs_worst_latency are the simulator's graph-based controller-switch "
    "delays; all controllers run on one host, so control-channel latency is not emulated."
)


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def sha256(path: Optional[Path]) -> Optional[str]:
    if path is None or not Path(path).exists():
        return None
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> Optional[str]:
    try:
        return subprocess.run(
            ["git", "-c", f"safe.directory={PROJECT_ROOT}", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def stats(values: List[float]) -> Dict[str, Optional[float]]:
    values = [float(v) for v in values if v is not None]
    if not values:
        return {"mean": None, "p95": None, "max": None, "n": 0}
    return {
        "mean": float(np.mean(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
        "n": len(values),
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Config JSON (default configs/mininet_os3e.json)")
    parser.add_argument("--profile", default=None, help="Profile in the config (synthetic | real_case_study)")
    parser.add_argument("--mode", choices=["current", "predictive"], default=None)
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=None)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--interval", type=int, default=None, help="Override timing.interval_s")
    parser.add_argument("--dry-run", action="store_true", help="No network; loads = trace rows")
    parser.add_argument("--dry-run-noise", type=float, default=0.0)
    parser.add_argument("--skip-simulator-reference", action="store_true")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args(argv)


def run_policy(*, policy, cfg, args, mode, trace, spec, spec_path, policy_dir, start, steps, seed) -> Dict:
    model_path = cfg.model_path(mode) if policy == "rdqn" else None
    gru_path = cfg.gru_model_path if mode == "predictive" else None
    env = make_simulator_env(
        cfg.topology, cfg.num_slave_controllers, cfg.num_parked_controllers,
        trace, mode, gru_path, max_steps=10**9,
    )
    shadow = ShadowState(env)
    selector = PolicySelector(policy, model_path, env, seed)
    shadow.reset(seed=seed, start_index=start)

    if args.dry_run:
        backend = DryRunBackend(noise_std=args.dry_run_noise, seed=seed)
    else:
        backend = LiveMininetBackend(cfg, spec, spec_path, policy_dir / "controllers", seed)
    executor = ActionExecutor(backend)

    steps_path = policy_dir / "emulated_steps.jsonl"
    records: List[Dict] = []
    measured_history: List[np.ndarray] = []
    forecasts: List[Optional[np.ndarray]] = []
    warmup_calibration: List[float] = []
    setup: Dict = {}
    interrupted = False

    print(f"[{policy}] starting {backend.name} backend "
          f"(state_dim={shadow.state_dim}, action_dim={shadow.action_dim})", flush=True)
    try:
        setup = backend.start(dict(env.switch_to_controller), set(env.active_slaves))
        for row in warmup_rows(trace, start, shadow.history_length):
            m = backend.run_interval(row)
            shadow.ingest(m.loads)
            warmup_calibration.append(float(np.mean(np.abs(m.loads - row))))

        with open(steps_path, "w", encoding="utf-8") as out:
            for k in range(steps):
                t_step = time.time()
                trace_index = start + k
                row = trace[trace_index]
                m = backend.run_interval(row)
                forecast = shadow.ingest(m.loads)
                state = shadow.observation()
                if state.shape[0] != shadow.state_dim:
                    raise RuntimeError(f"State has {state.shape[0]} values, expected {shadow.state_dim}")
                action, raw = selector.select(state, env)
                decision = shadow.apply(action, raw)
                diff = compute_diff(decision.mapping_before, decision.mapping_after,
                                    decision.active_before, decision.active_after)
                execution = executor.execute(diff) if not diff.empty else {"events": [], "all_success": True}
                actual = backend.assignment()
                consistent = all(actual.get(sw) == c for sw, c in decision.mapping_after.items())

                measured_history.append(m.loads.copy())
                forecasts.append(forecast)
                record = {
                    "step": k,
                    "trace_index": trace_index,
                    "action": decision.action,
                    "raw_action": decision.raw_action,
                    "action_name": decision.action_name,
                    "action_type": decision.action_type,
                    "success": decision.success,
                    "reward": decision.reward,
                    "info": {key: decision.info.get(key) for key in STEP_METRIC_KEYS + [
                        "pre_overloaded_count", "pre_underloaded_count"]},
                    "controller_loads": {str(c): float(v) for c, v in env.controller_loads.items()},
                    "active_after": sorted(decision.active_after),
                    "measured_loads": np.round(m.loads, 5).tolist(),
                    "trace_loads": np.round(row, 5).tolist(),
                    "calibration_mae": float(np.mean(np.abs(m.loads - row))),
                    "network_ops": execution["events"],
                    "network_ops_success": execution["all_success"],
                    "network_consistent": consistent,
                    "traffic": m.traffic_summary,
                    "controller_resources": m.controller_resources,
                    "switch_rates": m.switch_rates,
                    "step_wall_s": time.time() - t_step,
                }
                records.append(record)
                out.write(json.dumps(to_jsonable(record)) + "\n")
                out.flush()
                print(f"[{policy}] step {k + 1}/{steps} trace={trace_index} "
                      f"action={decision.action_name} ok={decision.success} "
                      f"net_ok={execution['all_success'] and consistent} "
                      f"cal_mae={record['calibration_mae']:.3f}", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        print(f"[{policy}] interrupted; writing partial results", flush=True)
    finally:
        backend.stop()

    horizon = getattr(env.traffic_predictor, "forecast_horizon", None)
    forecast_errors = []
    if horizon:
        for k, fc in enumerate(forecasts):
            if fc is not None and k + horizon < len(measured_history):
                forecast_errors.append(float(np.mean(np.abs(fc - measured_history[k + horizon]))))

    ops = [e for r in records for e in r["network_ops"]]
    traffic = [r["traffic"] for r in records if r["traffic"]]
    cpu = [res.get("cpu_percent") for r in records for res in r["controller_resources"].values()]
    rss = [res.get("rss_mb") for r in records for res in r["controller_resources"].values()]

    summary = {
        "policy": policy,
        "backend": backend.name,
        "observation_mode": mode,
        "interrupted": interrupted,
        "setup": setup,
        "simulator_style_metrics": summarize_decisions(records),
        "emulation_metrics": {
            "calibration_mae_measured_vs_trace": stats([r["calibration_mae"] for r in records]),
            "warmup_calibration_mae": stats(warmup_calibration),
            "loss_percent": stats([t.get("loss_percent") for t in traffic]),
            "jitter_ms": stats([t.get("jitter_ms") for t in traffic]),
            "rtt_avg_ms": stats([t.get("rtt_avg_ms") for t in traffic]),
            "rtt_max_ms": stats([t.get("rtt_max_ms") for t in traffic]),
            "offered_mbps": stats([t.get("offered_mbps") for t in traffic]),
            "received_mbps": stats([t.get("received_mbps") for t in traffic]),
            "failed_flows": int(sum(t.get("flows_failed", 0) for t in traffic)),
            "migration_handover_ms": stats([e.get("handover_ms") for e in ops if e["op"] == "migrate" and e.get("success")]),
            "flow_install_ms": stats([e.get("flow_install_ms") for e in ops if e["op"] == "migrate" and e.get("success")]),
            "controller_startup_ms": stats([e.get("startup_ms") for e in ops if e["op"] == "evoke" and e.get("success")]),
            "controller_stop_ms": stats([e.get("stop_ms") for e in ops if e["op"] == "park" and e.get("success")]),
            "network_ops_total": len(ops),
            "network_ops_failed": sum(1 for e in ops if not e.get("success")),
            "inconsistent_steps": sum(1 for r in records if not r["network_consistent"]),
            "controller_cpu_percent": stats(cpu),
            "controller_rss_mb": stats(rss),
            "forecast_mae_vs_measured": stats(forecast_errors),
        },
        "notes": {"energy": ENERGY_NOTE, "latency": LATENCY_NOTE},
    }
    return {"summary": summary, "records": records}


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, args.profile)
    run_cfg = cfg.section("run")
    if args.interval is not None:
        cfg.raw["timing"]["interval_s"] = int(args.interval)
    mode = args.mode or run_cfg.get("observation_mode", "current")
    policies = args.policies or run_cfg.get("policies", POLICIES)
    steps = int(args.steps if args.steps is not None else run_cfg["steps"])
    start = int(args.start_index if args.start_index is not None else run_cfg["start_index"])
    seed = int(args.seed if args.seed is not None else run_cfg["seed"])

    if not args.dry_run and os.geteuid() != 0:
        print("Live Mininet runs need root. Use:\n"
              "  sudo venv/bin/python -m mininet_integration.run_mininet_experiment ...\n"
              "or add --dry-run to test the pipeline without a network.", file=sys.stderr)
        return 2

    trace = np.load(cfg.traffic_path).astype(np.float32)
    if start < 0 or start + steps > len(trace):
        raise ValueError(f"Window [{start}, {start + steps}) is outside trace length {len(trace)}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = args.run_name or f"{stamp}_{cfg.topology}_{cfg.profile_name}_{mode}" + ("_dryrun" if args.dry_run else "")
    run_dir = assert_safe_output_dir(RESULTS_ROOT / run_id)
    run_dir.mkdir(parents=True, exist_ok=False)

    spec = build_topology_spec(cfg.topology, cfg.total_controllers)
    spec_path = spec.save(run_dir / "topology_spec.json")
    model_path = cfg.model_path(mode)
    gru_path = cfg.gru_model_path if mode == "predictive" else None
    (run_dir / "config_used.json").write_text(json.dumps(cfg.to_json(), indent=2))
    (run_dir / "run_info.json").write_text(json.dumps(to_jsonable({
        "run_id": run_id,
        "created": datetime.now().isoformat(),
        "backend": "dry_run" if args.dry_run else "mininet",
        "argv": sys.argv,
        "git_commit": git_commit(),
        "observation_mode": mode,
        "policies": policies,
        "start_index": start,
        "steps": steps,
        "seed": seed,
        "traffic_path": str(cfg.traffic_path),
        "traffic_sha256": sha256(cfg.traffic_path),
        "model_path": str(model_path),
        "model_sha256": sha256(model_path),
        "gru_model_path": str(gru_path) if gru_path else None,
        "gru_model_sha256": sha256(gru_path),
        "num_switches": spec.num_switches,
        "num_links": len(spec.links),
        "notes": {"energy": ENERGY_NOTE, "latency": LATENCY_NOTE},
    }), indent=2))

    comparison_rows = []
    try:
        for policy in policies:
            policy_dir = run_dir / policy
            policy_dir.mkdir()
            emu = run_policy(policy=policy, cfg=cfg, args=args, mode=mode, trace=trace, spec=spec,
                             spec_path=spec_path, policy_dir=policy_dir, start=start,
                             steps=steps, seed=seed)
            summary = emu["summary"]
            emu_metrics = summary["simulator_style_metrics"]
            comparison_rows.append({"policy": policy, "source": summary["backend"], **emu_metrics})

            if not args.skip_simulator_reference:
                n = len(emu["records"]) or steps
                sim_records = run_simulator_reference(
                    topology=cfg.topology,
                    num_slave_controllers=cfg.num_slave_controllers,
                    num_parked_controllers=cfg.num_parked_controllers,
                    trace=trace, observation_mode=mode, gru_model_path=gru_path,
                    model_path=model_path if policy == "rdqn" else None,
                    policy=policy, start_index=start, steps=n, seed=seed,
                )
                with open(policy_dir / "simulator_steps.jsonl", "w", encoding="utf-8") as out:
                    for rec in sim_records:
                        out.write(json.dumps(to_jsonable(rec)) + "\n")
                sim_metrics = summarize_decisions(sim_records)
                agree = [a["action"] == b["action"] for a, b in zip(emu["records"], sim_records)]
                summary["simulator_reference_metrics"] = sim_metrics
                summary["action_agreement_with_simulator"] = float(np.mean(agree)) if agree else None
                comparison_rows.append({"policy": policy, "source": "simulator", **sim_metrics})

            (policy_dir / "summary.json").write_text(json.dumps(to_jsonable(summary), indent=2))
            print(f"[{policy}] done: reward={emu_metrics.get('total_reward')} "
                  f"agreement={summary.get('action_agreement_with_simulator')}", flush=True)
            if summary["interrupted"]:
                break
    finally:
        if comparison_rows:
            fields = sorted({k for row in comparison_rows for k in row}, key=lambda k: (k not in ("policy", "source"), k))
            with open(run_dir / "comparison.csv", "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(comparison_rows)
            (run_dir / "comparison.json").write_text(json.dumps(to_jsonable(comparison_rows), indent=2))
        chown_to_invoking_user(RESULTS_ROOT)
        print(f"Results: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
