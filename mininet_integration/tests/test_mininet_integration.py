"""No-root tests for the Mininet integration.

    venv/bin/python -m unittest mininet_integration.tests.test_mininet_integration -v
"""

from __future__ import annotations

import json
import socket
import struct
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.dont_write_bytecode = True

import numpy as np  # noqa: E402

from mininet_integration._paths import PROJECT_ROOT, RESULTS_ROOT, assert_safe_output_dir  # noqa: E402
from mininet_integration.action_executor import (  # noqa: E402
    ActionExecutor,
    DryRunBackend,
    compute_diff,
)
from mininet_integration.config import load_config  # noqa: E402
from mininet_integration.mininet_baselines import run_simulator_reference  # noqa: E402
from mininet_integration.policy_adapter import PolicySelector, ShadowState, make_simulator_env  # noqa: E402
from mininet_integration.telemetry import CounterSnapshot, Telemetry  # noqa: E402
from mininet_integration.topology_zoo_mininet import TopologySpec, build_topology_spec  # noqa: E402
from mininet_integration.traffic_replay import (  # noqa: E402
    parse_iperf3_json,
    parse_ping,
    partner_indices,
    rates_for_row,
    warmup_rows,
)

CFG = load_config()
SPEC = build_topology_spec(CFG.topology, CFG.total_controllers)


class TopologySpecTests(unittest.TestCase):
    def test_matches_simulator_dimensions(self):
        env = make_simulator_env(CFG.topology, 3, 2, None, "current", None, 5)
        self.assertEqual(SPEC.num_switches, env.num_switches)
        self.assertEqual(sorted(len(c) for c in SPEC.components), [6, 28])

    def test_routes_are_loop_free_and_reach_destination(self):
        port_to_neighbor = {
            sw.index: {port: nb for nb, port in sw.neighbor_ports.items()} for sw in SPEC.switches
        }
        for comp in SPEC.components:
            for src in comp:
                for dst in comp:
                    ip = SPEC.switches[dst].host_ip
                    node, hops = src, 0
                    while True:
                        port = SPEC.routes[node][ip]
                        if port == SPEC.switches[node].host_port:
                            break
                        node = port_to_neighbor[node][port]
                        hops += 1
                        self.assertLess(hops, SPEC.num_switches, "routing loop")
                    self.assertEqual(node, dst)

    def test_no_route_across_disconnected_components(self):
        a, b = SPEC.components[0], SPEC.components[1]
        self.assertNotIn(SPEC.switches[b[0]].host_ip, SPEC.routes[a[0]])

    def test_json_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = SPEC.save(Path(tmp) / "spec.json")
            loaded = TopologySpec.load(path)
        self.assertEqual(loaded.to_json(), SPEC.to_json())


class TrafficTests(unittest.TestCase):
    def test_partners_are_permutation_within_components(self):
        partners = partner_indices(SPEC.num_switches, 17, SPEC.components)
        self.assertEqual(sorted(partners), list(range(SPEC.num_switches)))
        comp_of = {n: i for i, c in enumerate(SPEC.components) for n in c}
        for src, dst in enumerate(partners):
            self.assertNotEqual(src, dst)
            self.assertEqual(comp_of[src], comp_of[dst])

    def test_rates_and_warmup_padding(self):
        rates = rates_for_row(np.array([0.0, 0.001, 0.5, 1.2]), 20.0, 0.05)
        np.testing.assert_allclose(rates, [0.0, 0.0, 10.0, 20.0])
        trace = np.arange(20, dtype=np.float32).reshape(10, 2)
        rows = warmup_rows(trace, start=3, history_length=6)
        self.assertEqual(rows.shape, (5, 2))
        np.testing.assert_array_equal(rows[:3], np.repeat(trace[:1], 3, axis=0))
        np.testing.assert_array_equal(rows[3:], trace[1:3])
        self.assertEqual(warmup_rows(trace, 3, 1).shape[0], 0)

    def test_parsers(self):
        sample = {"end": {"sum": {"bits_per_second": 1e7, "jitter_ms": 0.2,
                                  "lost_packets": 3, "packets": 300, "lost_percent": 1.0}}}
        parsed = parse_iperf3_json(json.dumps(sample))
        self.assertTrue(parsed["ok"])
        self.assertAlmostEqual(parsed["received_mbps"], 10.0)
        self.assertFalse(parse_iperf3_json('{"error": "unable to connect"}')["ok"])
        ping = parse_ping("3 packets transmitted, 3 received, 0% packet loss\n"
                          "rtt min/avg/max/mdev = 10.1/12.5/15.0/2.0 ms")
        self.assertEqual(ping["rtt_avg_ms"], 12.5)
        self.assertEqual(ping["loss_percent"], 0.0)

    def test_proc_net_dev_single_read_snapshot(self):
        from mininet_integration.telemetry import parse_proc_net_dev

        text = (
            "Inter-|   Receive                                                |  Transmit\n"
            " face |bytes    packets errs drop fifo frame compressed multicast|bytes    packets errs drop fifo colls carrier compressed\n"
            "s1-eth1: 1000 10 0 2 0 0 0 0 5000 50 0 3 0 0 0 0\n"
            "s1-eth2: 300 3 0 0 0 0 0 0 700 7 0 0 0 0 0 0\n"
        )
        parsed = parse_proc_net_dev(text)
        self.assertEqual(parsed["s1-eth1"]["rx_bytes"], 1000)
        self.assertEqual(parsed["s1-eth1"]["tx_bytes"], 5000)
        self.assertEqual(parsed["s1-eth1"]["rx_dropped"], 2)
        self.assertEqual(parsed["s1-eth1"]["tx_dropped"], 3)
        snap = Telemetry(SPEC, CFG.section("traffic")).snapshot_from_text(text, 1.0)
        self.assertEqual(snap.host_port[0], 1000)
        real = parse_proc_net_dev(Path("/proc/net/dev").read_text())
        self.assertIn("lo", real)

    def test_telemetry_normalisation_removes_header_overhead(self):
        telemetry = Telemetry(SPEC, CFG.section("traffic"))
        n = SPEC.num_switches
        payload, overhead = 1200, 42
        target_mbps = np.linspace(0, 20, n)
        wire_bytes = target_mbps * 1e6 / 8 * (payload + overhead) / payload
        a = CounterSnapshot(0.0, np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n))
        b = CounterSnapshot(1.0, wire_bytes, wire_bytes, wire_bytes, np.zeros(n), np.zeros(n))
        loads = telemetry.normalised_loads(telemetry.rates(a, b))
        np.testing.assert_allclose(loads, target_mbps / 20.0, atol=1e-5)


class ShadowEquivalenceTests(unittest.TestCase):
    """With exact loads, the Mininet pipeline must reproduce the simulator."""

    def _run_shadow(self, policy, mode, start, steps, seed=7):
        trace = np.load(CFG.traffic_path).astype(np.float32)
        gru = CFG.gru_model_path if mode == "predictive" else None
        env = make_simulator_env(CFG.topology, 3, 2, trace, mode, gru, 10**9)
        shadow = ShadowState(env)
        model = CFG.model_path(mode) if policy == "rdqn" else None
        selector = PolicySelector(policy, model, env, seed)
        shadow.reset(seed, start)
        backend = DryRunBackend()
        backend.start(dict(env.switch_to_controller), set(env.active_slaves))
        executor = ActionExecutor(backend)
        for row in warmup_rows(trace, start, shadow.history_length):
            shadow.ingest(row)
        out = []
        for k in range(steps):
            shadow.ingest(trace[start + k])
            action, _ = selector.select(shadow.observation(), env)
            dec = shadow.apply(action)
            diff = compute_diff(dec.mapping_before, dec.mapping_after, dec.active_before, dec.active_after)
            self.assertTrue(executor.execute(diff)["all_success"])
            self.assertEqual(backend.assignment(), dec.mapping_after)
            self.assertEqual(backend.running, dec.active_after)
            out.append((action, round(dec.reward, 9)))
        sim = run_simulator_reference(
            topology=CFG.topology, num_slave_controllers=3, num_parked_controllers=2,
            trace=trace, observation_mode=mode, gru_model_path=gru, model_path=model,
            policy=policy, start_index=start, steps=steps, seed=seed,
        )
        return out, [(r["action"], round(r["reward"], 9)) for r in sim]

    def test_threshold_current(self):
        emu, sim = self._run_shadow("threshold", "current", 500, 60)
        self.assertEqual(emu, sim)

    def test_random_valid_current(self):
        emu, sim = self._run_shadow("random_valid", "current", 1200, 60)
        self.assertEqual(emu, sim)

    def test_rdqn_current(self):
        emu, sim = self._run_shadow("rdqn", "current", 0, 60)
        self.assertEqual(emu, sim)

    def test_rdqn_predictive(self):
        emu, sim = self._run_shadow("rdqn", "predictive", 5, 60)
        self.assertEqual(emu, sim)


class SafetyTests(unittest.TestCase):
    def test_output_dir_guard(self):
        for bad in ["models/x", "results/x", "experiments/results/x", "data/x", "/tmp/x"]:
            with self.assertRaises(ValueError):
                assert_safe_output_dir((PROJECT_ROOT / bad) if not bad.startswith("/") else Path(bad))
        self.assertTrue(str(assert_safe_output_dir(RESULTS_ROOT / "ok")).endswith("mininet_results/ok"))


# --------------------------------------------------------------------------
# os-ken controller handshake with a fake OpenFlow 1.3 switch (no root)
# --------------------------------------------------------------------------
OFP_HEADER = struct.Struct("!BBHI")


def _recv_msg(sock):
    header = b""
    while len(header) < 8:
        chunk = sock.recv(8 - len(header))
        if not chunk:
            return None
        header += chunk
    version, msg_type, length, xid = OFP_HEADER.unpack(header)
    body = b""
    while len(body) < length - 8:
        chunk = sock.recv(length - 8 - len(body))
        if not chunk:
            return None
        body += chunk
    return msg_type, xid, body


def fake_switch(port, dpid, stop_after_barrier=True, timeout=15.0):
    """Minimal OF1.3 switch: answers HELLO/FEATURES/PORT_DESC/ECHO/BARRIER."""
    sock = socket.create_connection(("127.0.0.1", port), timeout=timeout)
    sock.sendall(OFP_HEADER.pack(4, 0, 8, 1))  # HELLO
    flow_mods = 0
    deadline = time.time() + timeout
    try:
        while time.time() < deadline:
            msg = _recv_msg(sock)
            if msg is None:
                break
            msg_type, xid, body = msg
            if msg_type == 5:  # FEATURES_REQUEST
                payload = struct.pack("!QIBB2xII", dpid, 0, 254, 0, 0, 0)
                sock.sendall(OFP_HEADER.pack(4, 6, 8 + len(payload), xid) + payload)
            elif msg_type == 18:  # MULTIPART_REQUEST -> empty reply of same type
                mp_type = struct.unpack("!H", body[:2])[0]
                payload = struct.pack("!HH4x", mp_type, 0)
                sock.sendall(OFP_HEADER.pack(4, 19, 8 + len(payload), xid) + payload)
            elif msg_type == 2:  # ECHO_REQUEST
                sock.sendall(OFP_HEADER.pack(4, 3, 8 + len(body), xid) + body)
            elif msg_type == 14:  # FLOW_MOD
                flow_mods += 1
            elif msg_type == 20:  # BARRIER_REQUEST
                sock.sendall(OFP_HEADER.pack(4, 21, 8, xid))
                if stop_after_barrier:
                    time.sleep(0.3)
                    break
    finally:
        sock.close()
    return flow_mods


class ControllerHandshakeTests(unittest.TestCase):
    def test_controller_installs_routes_and_logs_flows_ready(self):
        from mininet_integration.controller_manager import ControllerManager, port_is_open

        port = 16653
        if port_is_open("127.0.0.1", port):
            self.skipTest(f"port {port} busy")
        with tempfile.TemporaryDirectory() as tmp:
            spec_path = SPEC.save(Path(tmp) / "spec.json")
            manager = ControllerManager(spec_path, Path(tmp) / "logs", base_port=port)
            try:
                start = manager.start(1)
                self.assertTrue(start["success"], start)
                reader = manager.reader(1)
                t0 = time.time()
                switch = SPEC.switches[3]
                result = {}
                thread = threading.Thread(target=lambda: result.update(n=fake_switch(port, switch.dpid)))
                thread.start()
                thread.join(20)
                ready = None
                for _ in range(100):
                    ready = reader.find("flows_ready", t0, switch.dpid)
                    if ready:
                        break
                    time.sleep(0.05)
                self.assertIsNotNone(ready, "no flows_ready event")
                expected = len(SPEC.routes[switch.index]) + 1  # routes + table-miss
                self.assertEqual(result.get("n"), expected)
                self.assertEqual(ready["flows"], expected)
                self.assertIsNotNone(reader.find("switch_connected", t0, switch.dpid))
            finally:
                manager.stop_all()
            self.assertFalse(manager.is_running(1))


if __name__ == "__main__":
    unittest.main()
