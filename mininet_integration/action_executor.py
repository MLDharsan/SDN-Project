"""Translate a simulator decision into real OVS / os-ken operations.

The shadow simulator decides *what* the new controller state is; this module
makes the network match it, in a safe order:

1. evoke   -> start the parked os-ken process and wait until it listens
2. migrate -> ``ovs-vsctl set-controller s<i> tcp:127.0.0.1:<port>`` and wait
              for the target controller's ``flows_ready`` event for that dpid
3. park    -> only after all of its switches have moved, stop the process

If a migration away from a controller fails, that controller is *not* parked
(its switches would be orphaned) and the step is flagged ``network_consistent
= False`` so the divergence is visible in the results.

Two backends implement the same interface:
- ``LiveMininetBackend`` (root, real network)
- ``DryRunBackend`` (no root; loads = trace rows; used for testing and for
  proving the shadow pipeline reproduces the simulator exactly)
"""

from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np


@dataclass
class MappingDiff:
    evoked: List[int] = field(default_factory=list)
    parked: List[int] = field(default_factory=list)
    migrations: List[Tuple[int, int, int]] = field(default_factory=list)  # (switch, src, dst)

    @property
    def empty(self) -> bool:
        return not (self.evoked or self.parked or self.migrations)


def compute_diff(
    mapping_before: Dict[int, int],
    mapping_after: Dict[int, int],
    active_before: Set[int],
    active_after: Set[int],
) -> MappingDiff:
    return MappingDiff(
        evoked=sorted(active_after - active_before),
        parked=sorted(active_before - active_after),
        migrations=[
            (sw, mapping_before[sw], mapping_after[sw])
            for sw in sorted(mapping_after)
            if mapping_before.get(sw) != mapping_after[sw]
        ],
    )


class ActionExecutor:
    def __init__(self, backend):
        self.backend = backend

    def execute(self, diff: MappingDiff) -> Dict:
        events: List[Dict] = []
        failed_sources: Set[int] = set()
        for cid in diff.evoked:
            rec = self.backend.start_controller(cid)
            events.append({"op": "evoke", **rec})
        for sw, src, dst in diff.migrations:
            rec = self.backend.migrate_switch(sw, dst)
            events.append({"op": "migrate", "switch": sw, "from": src, "to": dst, **rec})
            if not rec.get("success"):
                failed_sources.add(src)
        for cid in diff.parked:
            if cid in failed_sources:
                events.append({"op": "park", "controller_id": cid, "success": False,
                               "error": "skipped: switch migration away from it failed"})
                continue
            rec = self.backend.stop_controller(cid)
            events.append({"op": "park", **rec})
        return {
            "events": events,
            "all_success": all(e.get("success", False) for e in events),
        }


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------
@dataclass
class IntervalMeasurement:
    loads: np.ndarray
    traffic_summary: Dict = field(default_factory=dict)
    switch_rates: Dict = field(default_factory=dict)
    controller_resources: Dict = field(default_factory=dict)
    raw_traffic: Dict = field(default_factory=dict)


class DryRunBackend:
    """No network. Measured loads = trace row (+ optional Gaussian noise)."""

    name = "dry_run"

    def __init__(self, noise_std: float = 0.0, seed: int = 0):
        self.noise_std = noise_std
        self.rng = np.random.default_rng(seed)
        self.mapping: Dict[int, int] = {}
        self.running: Set[int] = set()

    def start(self, mapping: Dict[int, int], active: Set[int]) -> Dict:
        self.running = set(active)
        self.mapping = dict(mapping)
        return {"success": True, "backend": self.name}

    def run_interval(self, row: np.ndarray) -> IntervalMeasurement:
        loads = np.asarray(row, dtype=np.float32).copy()
        if self.noise_std > 0:
            loads = np.clip(loads + self.rng.normal(0, self.noise_std, loads.shape), 0, 1)
        return IntervalMeasurement(loads=loads.astype(np.float32))

    def start_controller(self, cid: int) -> Dict:
        self.running.add(cid)
        return {"controller_id": cid, "success": True, "startup_ms": 0.0}

    def stop_controller(self, cid: int) -> Dict:
        self.running.discard(cid)
        return {"controller_id": cid, "success": True, "stop_ms": 0.0}

    def migrate_switch(self, sw: int, cid: int) -> Dict:
        if cid not in self.running:
            return {"success": False, "error": f"controller {cid} not running"}
        self.mapping[sw] = cid
        return {"success": True, "handover_ms": 0.0}

    def assignment(self) -> Dict[int, Optional[int]]:
        return dict(self.mapping)

    def stop(self) -> None:
        pass


class LiveMininetBackend:
    """Real Mininet + OVS + os-ken network. Requires root."""

    name = "mininet"

    def __init__(self, cfg, spec, spec_path, log_dir, seed: int):
        from mininet_integration.controller_manager import ControllerManager
        from mininet_integration.telemetry import Telemetry

        self.cfg = cfg
        self.spec = spec
        self.ctrl_cfg = cfg.section("controllers")
        self.timing = cfg.section("timing")
        self.traffic_cfg = cfg.section("traffic")
        self.seed = seed
        self.controllers = ControllerManager(
            spec_path=spec_path,
            log_dir=log_dir,
            listen_host=self.ctrl_cfg["listen_host"],
            base_port=int(self.ctrl_cfg["base_port"]),
            startup_timeout_s=float(self.ctrl_cfg["startup_timeout_s"]),
        )
        self.telemetry = Telemetry(spec, self.traffic_cfg)
        self.net = None
        self.traffic = None
        self.mapping: Dict[int, int] = {}

    # ---------- lifecycle ----------
    def start(self, mapping: Dict[int, int], active: Set[int]) -> Dict:
        from mininet_integration.topology_zoo_mininet import build_mininet, start_mininet
        from mininet_integration.traffic_replay import TrafficReplayer

        t0 = time.time()
        startup = {cid: self.controllers.start(cid) for cid in sorted(active)}
        if not all(r["success"] for r in startup.values()):
            raise RuntimeError(f"Controller startup failed: {startup}")
        self.net = build_mininet(self.spec, self.cfg.section("network"))
        start_mininet(self.net)
        handovers = {sw: self.migrate_switch(sw, cid) for sw, cid in sorted(mapping.items())}
        failed = [sw for sw, r in handovers.items() if not r["success"]]
        if failed:
            raise RuntimeError(f"Initial controller assignment failed for switches {failed}")
        ping_loss = None
        if self.cfg.section("network").get("pingall_on_start", True):
            ping_loss = self.ping_components()
            if ping_loss > 0:
                raise RuntimeError(f"ping check lost {ping_loss}% after setup; routing is broken")
        self.traffic = TrafficReplayer(
            self.net, self.spec, self.traffic_cfg, self.cfg.section("probes"), seed=self.seed
        )
        self.traffic.start_servers()
        time.sleep(float(self.timing.get("settle_after_start_s", 2.0)))
        return {
            "success": True,
            "backend": self.name,
            "setup_s": time.time() - t0,
            "controller_startup": startup,
            "initial_handover_ms": [r["handover_ms"] for r in handovers.values()],
            "pingall_loss_percent": ping_loss,
        }

    def ping_components(self) -> float:
        """All-pairs ping inside each connected component (cross-island pairs are unroutable)."""
        hosts_by_index = {sw.index: self.net.get(sw.host_name) for sw in self.spec.switches}
        extra = [self.net.get(h[0]) for h in getattr(self.spec, "_extra_hosts", [])]
        sent = lost = 0.0
        for comp in self.spec.components or [list(hosts_by_index)]:
            hosts = [hosts_by_index[i] for i in comp] + (extra if comp == self.spec.components[0] else [])
            if len(hosts) < 2:
                continue
            pairs = len(hosts) * (len(hosts) - 1)
            lost += self.net.ping(hosts=hosts, timeout="1") / 100.0 * pairs
            sent += pairs
        return 100.0 * lost / sent if sent else 0.0

    def stop(self) -> None:
        try:
            if self.traffic is not None:
                self.traffic.stop_servers()
        finally:
            try:
                if self.net is not None:
                    self.net.stop()
            finally:
                self.controllers.stop_all()

    # ---------- traffic interval ----------
    def run_interval(self, row: np.ndarray) -> IntervalMeasurement:
        from mininet_integration.traffic_replay import rates_for_row, summarize_interval_traffic

        duration = int(self.timing["interval_s"])
        warmup = float(self.timing["measure_warmup_s"])
        margin = float(self.timing["measure_margin_s"])
        rates = rates_for_row(row, self.traffic_cfg["max_rate_mbps"], self.traffic_cfg["min_rate_mbps"])
        t_start = time.time()
        handles = self.traffic.start_interval(rates, duration)
        self.telemetry.controller_resources(self.controllers.running_pids())  # cpu baseline
        time.sleep(warmup)
        snap_a = self.telemetry.snapshot()
        time.sleep(max(t_start + duration - margin - time.time(), 0.1))
        snap_b = self.telemetry.snapshot()
        resources = self.telemetry.controller_resources(self.controllers.running_pids())
        raw = self.traffic.collect(handles, timeout_s=duration + 10)
        switch_rates = self.telemetry.rates(snap_a, snap_b)
        return IntervalMeasurement(
            loads=self.telemetry.normalised_loads(switch_rates),
            traffic_summary=summarize_interval_traffic(raw),
            switch_rates={k: np.asarray(v).round(4).tolist() for k, v in switch_rates.items()},
            controller_resources=resources,
            raw_traffic=raw,
        )

    # ---------- control-plane operations ----------
    def start_controller(self, cid: int) -> Dict:
        return self.controllers.start(cid)

    def stop_controller(self, cid: int) -> Dict:
        owned = [sw for sw, c in self.mapping.items() if c == cid]
        if owned:
            return {"controller_id": cid, "success": False,
                    "error": f"still owns switches {owned}; refusing to stop"}
        return self.controllers.stop(cid)

    def migrate_switch(self, sw: int, cid: int) -> Dict:
        if not self.controllers.is_running(cid):
            return {"success": False, "error": f"controller {cid} not running"}
        switch = self.spec.switches[sw]
        target = self.controllers.target_for(cid)
        reader = self.controllers.reader(cid)
        t0 = time.time()
        proc = subprocess.run(
            ["ovs-vsctl", "set-controller", switch.name, target],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            return {"success": False, "error": proc.stderr.strip()}
        deadline = t0 + float(self.ctrl_cfg["handover_timeout_s"])
        connected = ready = None
        while time.time() < deadline:
            connected = connected or reader.find("switch_connected", t0, switch.dpid)
            ready = reader.find("flows_ready", t0, switch.dpid)
            if ready:
                break
            time.sleep(0.01)
        if not ready:
            return {"success": False, "error": "handover timeout", "target": target,
                    "handover_ms": (time.time() - t0) * 1000.0}
        self.mapping[sw] = cid
        return {
            "success": True,
            "target": target,
            "connect_ms": (connected["t"] - t0) * 1000.0 if connected else None,
            "flow_install_ms": ready.get("flow_install_ms"),
            "handover_ms": (ready["t"] - t0) * 1000.0,
        }

    def assignment(self) -> Dict[int, Optional[int]]:
        """Read the real switch->controller mapping back from OVSDB."""
        port_to_cid = {self.controllers.target_for(c): c for c in range(1, self.cfg.total_controllers)}
        actual = self.telemetry.ovs_assignments()
        result = {}
        for sw in self.spec.switches:
            targets = actual.get(sw.name, {}).get("targets", [])
            result[sw.index] = port_to_cid.get(targets[0]) if len(targets) == 1 else None
        return result
