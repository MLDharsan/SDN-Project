"""Live Mininet/OVS telemetry, converted into RDQN-compatible features.

State compatibility rule: the trained RDQN only ever saw *normalised traffic
load* features. So the measured byte rate entering each switch from its host
is converted back to [0, 1] with the same ``max_rate_mbps`` used by the replay
(after removing Ethernet/IP/UDP header overhead). Controller loads are then
computed by the unchanged simulator code (mean of assigned switch loads), so
controller CPU is *reported* but never fed to the model.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

PROC_NET_DEV = Path("/proc/net/dev")
CLOCK_TICKS = os.sysconf("SC_CLK_TCK")
_ZERO = {"rx_bytes": 0, "rx_packets": 0, "rx_dropped": 0, "tx_bytes": 0, "tx_packets": 0, "tx_dropped": 0}


def parse_proc_net_dev(text: str) -> Dict[str, Dict[str, int]]:
    """Parse /proc/net/dev: all interface counters from ONE read.

    Reading every counter in a single syscall gives one consistent timestamp
    for all switches. (Reading hundreds of sysfs files one by one under load
    skews per-switch time windows by hundreds of milliseconds.)
    """
    result = {}
    for line in text.splitlines()[2:]:
        if ":" not in line:
            continue
        name, data = line.split(":", 1)
        f = data.split()
        if len(f) < 12:
            continue
        result[name.strip()] = {
            "rx_bytes": int(f[0]), "rx_packets": int(f[1]), "rx_dropped": int(f[3]),
            "tx_bytes": int(f[8]), "tx_packets": int(f[9]), "tx_dropped": int(f[11]),
        }
    return result


@dataclass
class CounterSnapshot:
    t: float
    host_port: np.ndarray       # rx_bytes on each switch's host-facing port
    total_rx: np.ndarray        # rx_bytes summed over all switch ports
    total_tx: np.ndarray
    rx_packets: np.ndarray
    dropped: np.ndarray


class Telemetry:
    def __init__(self, spec, traffic_cfg: Dict):
        self.spec = spec
        self.max_rate_mbps = float(traffic_cfg["max_rate_mbps"])
        payload = float(traffic_cfg.get("udp_payload_bytes", 1200))
        overhead = float(traffic_cfg.get("wire_overhead_bytes", 42))
        self.payload_fraction = payload / (payload + overhead)
        self.switch_ifaces: List[List[str]] = []
        for sw in spec.switches:
            ports = [sw.host_port] + sorted(sw.neighbor_ports.values())
            self.switch_ifaces.append([f"{sw.name}-eth{p}" for p in ports])
        self._cpu_prev: Dict[int, tuple] = {}

    # ---------------- switch counters ----------------
    def snapshot(self) -> CounterSnapshot:
        t_before = time.time()
        text = PROC_NET_DEV.read_text()
        t = (t_before + time.time()) / 2.0
        return self.snapshot_from_text(text, t)

    def snapshot_from_text(self, text: str, t: float) -> CounterSnapshot:
        counters = parse_proc_net_dev(text)
        n = self.spec.num_switches
        host = np.zeros(n)
        rx = np.zeros(n)
        tx = np.zeros(n)
        pk = np.zeros(n)
        dr = np.zeros(n)
        for i, ifaces in enumerate(self.switch_ifaces):
            for j, iface in enumerate(ifaces):
                c = counters.get(iface, _ZERO)
                if j == 0:
                    host[i] = c["rx_bytes"]
                rx[i] += c["rx_bytes"]
                tx[i] += c["tx_bytes"]
                pk[i] += c["rx_packets"]
                dr[i] += c["rx_dropped"] + c["tx_dropped"]
        return CounterSnapshot(t, host, rx, tx, pk, dr)

    def rates(self, a: CounterSnapshot, b: CounterSnapshot) -> Dict[str, np.ndarray]:
        dt = max(b.t - a.t, 1e-6)
        to_mbps = 8.0 / dt / 1e6
        return {
            "host_ingress_mbps": (b.host_port - a.host_port) * to_mbps,
            "switch_rx_mbps": (b.total_rx - a.total_rx) * to_mbps,
            "switch_tx_mbps": (b.total_tx - a.total_tx) * to_mbps,
            "switch_rx_pps": (b.rx_packets - a.rx_packets) / dt,
            "switch_drops": b.dropped - a.dropped,
            "window_s": np.array([dt]),
        }

    def normalised_loads(self, rates: Dict[str, np.ndarray]) -> np.ndarray:
        payload_mbps = rates["host_ingress_mbps"] * self.payload_fraction
        return np.clip(payload_mbps / self.max_rate_mbps, 0.0, 1.0).astype(np.float32)

    # ---------------- controller processes ----------------
    @staticmethod
    def _proc_ticks(pid: int) -> Optional[int]:
        try:
            fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
            return int(fields[11]) + int(fields[12])  # utime + stime
        except (OSError, IndexError, ValueError):
            return None

    @staticmethod
    def _proc_rss_mb(pid: int) -> Optional[float]:
        try:
            for line in Path(f"/proc/{pid}/status").read_text().splitlines():
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
        except (OSError, ValueError):
            return None
        return None

    def controller_resources(self, pids: Dict[int, int]) -> Dict[int, Dict]:
        now = time.time()
        result = {}
        for cid, pid in pids.items():
            ticks = self._proc_ticks(pid)
            cpu = None
            prev = self._cpu_prev.get(pid)
            if ticks is not None and prev is not None:
                cpu = 100.0 * (ticks - prev[1]) / CLOCK_TICKS / max(now - prev[0], 1e-6)
            if ticks is not None:
                self._cpu_prev[pid] = (now, ticks)
            result[cid] = {"pid": pid, "cpu_percent": cpu, "rss_mb": self._proc_rss_mb(pid)}
        return result

    # ---------------- OVS controller assignment ----------------
    @staticmethod
    def _ovs_json(args: List[str]) -> Dict:
        out = subprocess.run(
            ["ovs-vsctl", "--format=json"] + args,
            check=True, capture_output=True, text=True,
        ).stdout
        return json.loads(out)

    @staticmethod
    def _ovs_value(value):
        if isinstance(value, list) and value and value[0] == "set":
            return [Telemetry._ovs_value(v) for v in value[1]]
        if isinstance(value, list) and value and value[0] == "uuid":
            return value[1]
        return value

    def ovs_assignments(self) -> Dict[str, Dict]:
        """Return {bridge: {"targets": [...], "connected": bool}} straight from OVSDB."""
        controllers = self._ovs_json(["--columns=_uuid,target,is_connected", "list", "Controller"])
        by_uuid = {}
        for row in controllers["data"]:
            uuid, target, connected = (self._ovs_value(v) for v in row)
            by_uuid[uuid] = {"target": target, "connected": bool(connected)}
        bridges = self._ovs_json(["--columns=name,controller", "list", "Bridge"])
        result = {}
        names = {sw.name for sw in self.spec.switches}
        for name, ctrl in bridges["data"]:
            if name not in names:
                continue
            ctrl = self._ovs_value(ctrl)
            uuids = ctrl if isinstance(ctrl, list) else ([ctrl] if ctrl else [])
            entries = [by_uuid[u] for u in uuids if u in by_uuid]
            result[name] = {
                "targets": [e["target"] for e in entries],
                "connected": any(e["connected"] for e in entries),
            }
        return result
