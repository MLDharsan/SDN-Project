#!/usr/bin/env python3
"""Step-by-step Mininet smoke tests (implementation-order steps 2, 3 and 6).

    sudo venv/bin/python -m mininet_integration.smoke_test --topo single
    sudo venv/bin/python -m mininet_integration.smoke_test --topo linear3 --traffic
    sudo venv/bin/python -m mininet_integration.smoke_test --topo linear3 --migrate
    sudo venv/bin/python -m mininet_integration.smoke_test --topo os3e --traffic --migrate

Each stage prints PASS/FAIL. Logs go to mininet_results/smoke_<timestamp>/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

from mininet_integration._paths import RESULTS_ROOT, assert_safe_output_dir, chown_to_invoking_user  # noqa: E402
from mininet_integration.action_executor import LiveMininetBackend  # noqa: E402
from mininet_integration.config import load_config  # noqa: E402
from mininet_integration.topology_zoo_mininet import build_small_spec, build_topology_spec  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--topo", choices=["single", "linear3", "os3e"], default="single")
    parser.add_argument("--traffic", action="store_true", help="Replay a static 50%% load interval")
    parser.add_argument("--migrate", action="store_true", help="Move switch 0 to a second controller")
    parser.add_argument("--load", type=float, default=0.5)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    if os.geteuid() != 0:
        print("Mininet needs root: sudo venv/bin/python -m mininet_integration.smoke_test ...", file=sys.stderr)
        return 2

    cfg = load_config(args.config)
    spec = build_topology_spec(cfg.topology, cfg.total_controllers) if args.topo == "os3e" else build_small_spec(args.topo)
    out_dir = assert_safe_output_dir(RESULTS_ROOT / f"smoke_{args.topo}_{datetime.now():%Y%m%d_%H%M%S}")
    out_dir.mkdir(parents=True)
    spec_path = spec.save(out_dir / "topology_spec.json")
    cfg.raw["network"]["pingall_on_start"] = True

    backend = LiveMininetBackend(cfg, spec, spec_path, out_dir / "controllers", seed=0)
    results = {"topology": args.topo, "switches": spec.num_switches}
    ok = True
    try:
        mapping = {sw.index: 1 for sw in spec.switches}
        setup = backend.start(mapping, {1})
        results["setup"] = setup
        print(f"PASS  setup + ping check (loss {setup['pingall_loss_percent']}%) in {setup['setup_s']:.1f}s")

        if args.traffic:
            n = spec.num_switches
            row = np.full(n, args.load, dtype=np.float32)
            if args.topo == "single":
                print("SKIP  traffic: 'single' has only one switch-load entry; use linear3")
            else:
                m = backend.run_interval(row)
                mae = float(np.mean(np.abs(m.loads - row)))
                results["traffic"] = {"target": args.load, "measured": m.loads.tolist(),
                                      "calibration_mae": mae, "summary": m.traffic_summary}
                passed = mae < 0.05 and m.traffic_summary.get("flows_failed", 1) == 0
                ok &= passed
                print(f"{'PASS' if passed else 'FAIL'}  traffic replay: target={args.load} "
                      f"measured mean={m.loads.mean():.3f} mae={mae:.3f} "
                      f"loss={m.traffic_summary.get('loss_percent'):.2f}% "
                      f"rtt={m.traffic_summary.get('rtt_avg_ms')}")

        if args.migrate:
            start = backend.start_controller(2)
            rec = backend.migrate_switch(0, 2)
            actual = backend.assignment()
            loss = backend.ping_components()
            passed = start["success"] and rec["success"] and actual.get(0) == 2 and loss == 0
            ok &= passed
            results["migration"] = {"startup": start, "handover": rec, "assignment": actual,
                                    "pingall_loss_after": loss}
            print(f"{'PASS' if passed else 'FAIL'}  migrate s1 c1->c2: handover={rec.get('handover_ms')} ms, "
                  f"controller start={start.get('startup_ms'):.0f} ms, pingAll loss after={loss}%")
            back = backend.migrate_switch(0, 1)
            stop = backend.stop_controller(2)
            passed = back["success"] and stop["success"]
            ok &= passed
            print(f"{'PASS' if passed else 'FAIL'}  migrate back + park c2 (stop {stop.get('stop_ms', 0):.0f} ms)")
    except Exception as exc:  # report and clean up
        ok = False
        results["error"] = repr(exc)
        print(f"FAIL  {exc!r}")
    finally:
        backend.stop()
        (out_dir / "smoke_result.json").write_text(json.dumps(results, indent=2, default=str))
        chown_to_invoking_user(RESULTS_ROOT)
    print(f"{'ALL PASSED' if ok else 'SOME CHECKS FAILED'} - details in {out_dir}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
