"""Replay one traffic-trace row per decision interval as real host traffic.

Each trace value is a normalised switch load in [0, 1]. Host ``h{i+1}``
(attached to switch ``i``) sends UDP at ``load * max_rate_mbps`` to a fixed
partner host, so the bytes entering switch ``i`` from its host reproduce the
trace value. Partners form a permutation, so every host runs exactly one
iperf3 server and receives exactly one flow.
"""

from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

PING_RTT_RE = re.compile(r"=\s*([\d.]+)/([\d.]+)/([\d.]+)/([\d.]+)\s*ms")
PING_LOSS_RE = re.compile(r"([\d.]+)% packet loss")


def rates_for_row(row: np.ndarray, max_rate_mbps: float, min_rate_mbps: float) -> np.ndarray:
    rates = np.clip(np.asarray(row, dtype=np.float64), 0.0, 1.0) * float(max_rate_mbps)
    rates[rates < float(min_rate_mbps)] = 0.0
    return rates


def partner_indices(num_hosts: int, offset: int, components=None) -> List[int]:
    """Fixed destination per host: a cyclic shift *within* each connected component.

    Every host therefore receives exactly one flow (one iperf3 server each) and
    no flow targets an unreachable component. A singleton component sends to
    itself, which ``start_interval`` skips.
    """
    components = components or [list(range(num_hosts))]
    partners = list(range(num_hosts))
    for comp in components:
        comp = sorted(comp)
        size = len(comp)
        if size < 2:
            continue
        shift = offset % size or (size // 2)
        for pos, node in enumerate(comp):
            partners[node] = comp[(pos + shift) % size]
    return partners


def warmup_rows(trace: np.ndarray, start: int, history_length: int) -> np.ndarray:
    """Rows that precede ``start``, padded exactly like the simulator's history priming."""
    needed = history_length - 1
    if needed <= 0:
        return np.zeros((0, trace.shape[1]), dtype=np.float32)
    first = max(0, start - needed)
    prior = [row.copy() for row in trace[first:start]]
    pad = prior[0] if prior else trace[min(start, len(trace) - 1)].copy()
    rows = [pad.copy() for _ in range(needed - len(prior))] + prior
    return np.asarray(rows, dtype=np.float32)


def parse_iperf3_json(text: str) -> Dict:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"ok": False, "error": "invalid iperf3 json"}
    if "error" in data:
        return {"ok": False, "error": str(data["error"])}
    end = data.get("end", {})
    summary = end.get("sum_received") or end.get("sum") or {}
    sent = end.get("sum_sent") or end.get("sum") or {}
    return {
        "ok": True,
        "sent_mbps": float(sent.get("bits_per_second", 0.0)) / 1e6,
        "received_mbps": float(summary.get("bits_per_second", 0.0)) / 1e6,
        "jitter_ms": float(summary.get("jitter_ms", 0.0)),
        "lost_packets": int(summary.get("lost_packets", 0)),
        "packets": int(summary.get("packets", 0)),
        "lost_percent": float(summary.get("lost_percent", 0.0)),
    }


def parse_ping(text: str) -> Dict:
    rtt = PING_RTT_RE.search(text)
    loss = PING_LOSS_RE.search(text)
    return {
        "rtt_avg_ms": float(rtt.group(2)) if rtt else None,
        "rtt_max_ms": float(rtt.group(3)) if rtt else None,
        "loss_percent": float(loss.group(1)) if loss else None,
    }


@dataclass
class IntervalHandles:
    flows: List[Tuple[int, int, float, subprocess.Popen]]
    pings: List[Tuple[int, int, subprocess.Popen]]


class TrafficReplayer:
    def __init__(self, net, spec, traffic_cfg: Dict, probes_cfg: Dict, seed: int = 0):
        self.net = net
        self.spec = spec
        self.cfg = traffic_cfg
        self.probes_cfg = probes_cfg
        self.port = int(traffic_cfg.get("iperf3_port", 5201))
        self.partners = partner_indices(
            spec.num_switches, int(traffic_cfg.get("pairing_offset", 1)), spec.components
        )
        self.servers: List[subprocess.Popen] = []
        rng = np.random.default_rng(seed)
        candidates = [
            (a, b)
            for comp in (spec.components or [list(range(spec.num_switches))])
            for a in comp for b in comp if a != b
        ]
        wanted = min(int(probes_cfg.get("ping_pairs", 0)), len(candidates))
        chosen = rng.choice(len(candidates), size=wanted, replace=False) if wanted else []
        self.ping_pairs = sorted(candidates[int(i)] for i in chosen)

    def host(self, index: int):
        return self.net.get(self.spec.switches[index].host_name)

    def start_servers(self) -> None:
        for index in sorted(set(self.partners)):
            self.servers.append(
                self.host(index).popen(
                    ["iperf3", "-s", "-p", str(self.port)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            )

    def stop_servers(self) -> None:
        for proc in self.servers:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
        self.servers.clear()

    def start_interval(self, rates_mbps: Sequence[float], duration_s: int) -> IntervalHandles:
        payload = int(self.cfg.get("udp_payload_bytes", 1200))
        flows = []
        for src, rate in enumerate(rates_mbps):
            dst = self.partners[src]
            if rate <= 0 or dst == src:
                continue
            dst_ip = self.spec.switches[dst].host_ip
            cmd = [
                "iperf3", "-c", dst_ip, "-p", str(self.port), "-u",
                "-b", f"{rate:.4f}M", "-l", str(payload), "-t", str(int(duration_s)), "-J",
            ]
            flows.append(
                (src, dst, float(rate),
                 self.host(src).popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      universal_newlines=True))
            )
        pings = []
        count = int(self.probes_cfg.get("ping_count", 3))
        for src, dst in self.ping_pairs:
            cmd = ["ping", "-q", "-c", str(count), "-i", "0.2", "-W", "1",
                   self.spec.switches[dst].host_ip]
            pings.append(
                (src, dst, self.host(src).popen(cmd, stdout=subprocess.PIPE,
                                                stderr=subprocess.STDOUT,
                                                universal_newlines=True))
            )
        return IntervalHandles(flows, pings)

    @staticmethod
    def collect(handles: IntervalHandles, timeout_s: float) -> Dict:
        flow_results = []
        for src, dst, rate, proc in handles.flows:
            try:
                out, err = proc.communicate(timeout=timeout_s)
                result = parse_iperf3_json(out)
                if not result["ok"] and err:
                    result["stderr"] = err.strip()[-300:]
            except subprocess.TimeoutExpired:
                proc.kill()
                result = {"ok": False, "error": "iperf3 timeout"}
            result.update({"src": src, "dst": dst, "target_mbps": rate})
            flow_results.append(result)
        ping_results = []
        for src, dst, proc in handles.pings:
            try:
                out, _ = proc.communicate(timeout=timeout_s)
                result = parse_ping(out)
            except subprocess.TimeoutExpired:
                proc.kill()
                result = {"rtt_avg_ms": None, "rtt_max_ms": None, "loss_percent": None}
            result.update({"src": src, "dst": dst})
            ping_results.append(result)
        return {"flows": flow_results, "pings": ping_results}


def summarize_interval_traffic(traffic: Dict) -> Dict[str, Optional[float]]:
    flows = [f for f in traffic.get("flows", []) if f.get("ok")]
    pings = [p for p in traffic.get("pings", []) if p.get("rtt_avg_ms") is not None]
    total_packets = sum(f["packets"] for f in flows)
    lost = sum(f["lost_packets"] for f in flows)
    return {
        "flows_ok": len(flows),
        "flows_failed": len(traffic.get("flows", [])) - len(flows),
        "offered_mbps": float(sum(f.get("target_mbps", 0.0) for f in traffic.get("flows", []))),
        "received_mbps": float(sum(f["received_mbps"] for f in flows)),
        "loss_percent": (100.0 * lost / total_packets) if total_packets else 0.0,
        "jitter_ms": float(np.mean([f["jitter_ms"] for f in flows])) if flows else None,
        "rtt_avg_ms": float(np.mean([p["rtt_avg_ms"] for p in pings])) if pings else None,
        "rtt_max_ms": float(np.max([p["rtt_max_ms"] for p in pings])) if pings else None,
    }
