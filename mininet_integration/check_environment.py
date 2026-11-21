#!/usr/bin/env python3
"""Pre-flight check (no root needed, changes nothing).

    venv/bin/python -m mininet_integration.check_environment

Checks binaries, Python imports, the OS3E topology spec, checkpoint state and
action sizes (46/86 and 45), the traffic trace, and that controller ports
are free.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

from mininet_integration import _paths  # noqa: E402
from mininet_integration.config import load_config  # noqa: E402

results = []


def check(name, fn):
    try:
        detail = fn()
        results.append((True, name, detail or ""))
    except Exception as exc:  # noqa: BLE001
        results.append((False, name, str(exc)))


def binary(name):
    def _fn():
        path = shutil.which(name)
        if not path:
            raise FileNotFoundError(f"{name} not on PATH")
        return path
    return _fn


def main() -> int:
    cfg = load_config()
    for name in ("mn", "mnexec", "ovs-vsctl", "ovs-ofctl", "iperf3", "ping", "tc"):
        check(f"binary {name}", binary(name))

    def ovs_running():
        out = subprocess.run(["ovs-vsctl", "--version"], capture_output=True, text=True, check=True)
        return out.stdout.splitlines()[0]
    check("open vswitch", ovs_running)

    def mininet_import():
        _paths.ensure_mininet_importable()
        import mininet
        from mininet.net import Mininet  # noqa: F401
        return mininet.__file__
    check("mininet python import", mininet_import)

    def osken():
        from mininet_integration.controller_manager import find_osken_manager
        import os_ken
        return f"os_ken {getattr(os_ken, '__version__', '?')} / {find_osken_manager()}"
    check("os-ken controller", osken)

    def spec():
        from mininet_integration.topology_zoo_mininet import build_topology_spec
        s = build_topology_spec(cfg.topology, cfg.total_controllers)
        return f"{s.num_switches} switches, {len(s.links)} links"
    check("topology spec", spec)

    def trace():
        data = np.load(cfg.traffic_path)
        return f"{cfg.traffic_path.name} shape={data.shape} range=[{data.min():.3f}, {data.max():.3f}]"
    check(f"traffic trace ({cfg.profile_name})", trace)

    for mode in ("current", "predictive"):
        def model(mode=mode):
            from mininet_integration.policy_adapter import PolicySelector, make_simulator_env
            data = np.load(cfg.traffic_path).astype(np.float32)
            env = make_simulator_env(cfg.topology, cfg.num_slave_controllers, cfg.num_parked_controllers,
                                     data, mode, cfg.gru_model_path if mode == "predictive" else None, 10)
            sel = PolicySelector("rdqn", cfg.model_path(mode), env, 0)
            return (f"{cfg.model_path(mode).name}: state_dim={sel.checkpoint['state_dim']} "
                    f"action_dim={sel.checkpoint['action_dim']} (env {env.observation_space.shape[0]}/"
                    f"{env.action_space.n})")
        check(f"RDQN checkpoint {mode}", model)

    def ports():
        from mininet_integration.controller_manager import port_is_open
        c = cfg.section("controllers")
        busy = [p for p in range(c["base_port"], c["base_port"] + cfg.total_controllers - 1)
                if port_is_open(c["listen_host"], p)]
        if busy:
            raise RuntimeError(f"ports already in use: {busy}")
        return f"{c['base_port']}-{c['base_port'] + cfg.total_controllers - 2} free"
    check("controller ports", ports)

    for ok, name, detail in results:
        print(f"{'OK  ' if ok else 'FAIL'}  {name:32s} {detail}")
    failed = [r for r in results if not r[0]]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
