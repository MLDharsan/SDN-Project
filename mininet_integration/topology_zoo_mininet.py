"""Build an OS3E (Topology Zoo) Mininet network from the simulator's topology.

The *same* ``TopologyLoader`` used by the simulator supplies the graph, the
node ordering (switch index ``i`` here == switch index ``i`` in the RDQN state)
and the geographic per-link delays.

Two layers:
- ``build_topology_spec`` is pure Python (no root) and produces a JSON-able
  description: switches, hosts, port numbers, link delays and loop-free
  shortest-path routes. The os-ken controllers read this same file.
- ``build_mininet`` turns a spec into a running Mininet network (needs root).

Every switch gets exactly one host on port 1. Inter-switch links use ports
2, 3, ... in deterministic order. OS3E contains cycles, so routing is
proactive shortest-path (by delay) per destination IP; no flooding is ever
used, which avoids broadcast storms without needing STP.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

import networkx as nx

from mininet_integration import _paths  # noqa: F401  (sets sys.path)
from utils.topology_loader import TopologyLoader

HOST_PORT = 1


@dataclass
class SwitchSpec:
    index: int
    name: str
    dpid: int
    label: str
    host_name: str
    host_ip: str
    host_mac: str
    host_port: int = HOST_PORT
    neighbor_ports: Dict[int, int] = field(default_factory=dict)

    @property
    def host_iface(self) -> str:
        return f"{self.name}-eth{self.host_port}"


@dataclass
class LinkSpec:
    a: int
    b: int
    port_a: int
    port_b: int
    delay_ms: float


@dataclass
class TopologySpec:
    name: str
    switches: List[SwitchSpec]
    links: List[LinkSpec]
    # routes[switch_index][dst_ip] = output port on that switch
    routes: Dict[int, Dict[str, int]]
    controller_host_nodes: List[int]
    # Connected components of the data plane (switch indices). The OS3E GML
    # in data/topologies is disconnected (28 + 6 nodes), so traffic and pings
    # are only generated inside a component.
    components: List[List[int]] = field(default_factory=list)

    @property
    def num_switches(self) -> int:
        return len(self.switches)

    def switch_by_dpid(self) -> Dict[int, SwitchSpec]:
        return {item.dpid: item for item in self.switches}

    def to_json(self) -> Dict:
        return {
            "name": self.name,
            "switches": [asdict(item) for item in self.switches],
            "links": [asdict(item) for item in self.links],
            "routes": {str(k): v for k, v in self.routes.items()},
            "controller_host_nodes": list(self.controller_host_nodes),
            "components": [list(c) for c in self.components],
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), indent=2))
        return path

    @classmethod
    def from_json(cls, data: Dict) -> "TopologySpec":
        switches = []
        for item in data["switches"]:
            item = dict(item)
            item["neighbor_ports"] = {int(k): int(v) for k, v in item["neighbor_ports"].items()}
            switches.append(SwitchSpec(**item))
        return cls(
            name=data["name"],
            switches=switches,
            links=[LinkSpec(**item) for item in data["links"]],
            routes={int(k): {ip: int(p) for ip, p in v.items()} for k, v in data["routes"].items()},
            controller_host_nodes=list(data["controller_host_nodes"]),
            components=[list(c) for c in data.get("components", [])],
        )

    @classmethod
    def load(cls, path: Path) -> "TopologySpec":
        return cls.from_json(json.loads(Path(path).read_text()))


def host_ip(index: int) -> str:
    number = index + 1
    return f"10.0.{number // 256}.{number % 256}"


def host_mac(index: int) -> str:
    number = index + 1
    return ":".join(f"{(number >> shift) & 0xFF:02x}" for shift in range(40, -1, -8))


def build_topology_spec(topology_name: str = "os3e", total_controllers: int = 6) -> TopologySpec:
    loader = TopologyLoader()
    topo, _delay_matrix, controller_hosts = loader.build_switch_to_controller_delay_matrix(
        topology_name, total_controllers
    )
    ordered = list(topo.ordered_nodes)
    index_of = {node: idx for idx, node in enumerate(ordered)}

    switches = [
        SwitchSpec(
            index=idx,
            name=f"s{idx + 1}",
            dpid=idx + 1,
            label=topo.node_labels.get(node, str(node)),
            host_name=f"h{idx + 1}",
            host_ip=host_ip(idx),
            host_mac=host_mac(idx),
        )
        for idx, node in enumerate(ordered)
    ]

    next_port = {idx: HOST_PORT + 1 for idx in range(len(ordered))}
    links: List[LinkSpec] = []
    edges = sorted(tuple(sorted((index_of[u], index_of[v]))) for u, v in topo.graph.edges())
    for a, b in edges:
        delay = float(topo.graph[ordered[a]][ordered[b]].get("delay_ms", 1.0))
        port_a, port_b = next_port[a], next_port[b]
        next_port[a] += 1
        next_port[b] += 1
        switches[a].neighbor_ports[b] = port_a
        switches[b].neighbor_ports[a] = port_b
        links.append(LinkSpec(a=a, b=b, port_a=port_a, port_b=port_b, delay_ms=delay))

    graph = nx.Graph()
    graph.add_nodes_from(range(len(ordered)))
    for link in links:
        graph.add_edge(link.a, link.b, delay_ms=link.delay_ms)
    components = sorted((sorted(c) for c in nx.connected_components(graph)), key=lambda c: c[0])

    routes: Dict[int, Dict[str, int]] = {}
    for src in range(len(ordered)):
        paths = nx.single_source_dijkstra_path(graph, src, weight="delay_ms")
        table: Dict[str, int] = {}
        for dst, path in paths.items():  # only reachable destinations
            if dst == src:
                table[switches[dst].host_ip] = switches[src].host_port
            else:
                table[switches[dst].host_ip] = switches[src].neighbor_ports[path[1]]
        routes[src] = table

    return TopologySpec(
        name=topology_name,
        switches=switches,
        links=links,
        routes=routes,
        controller_host_nodes=[index_of[node] for node in controller_hosts],
        components=components,
    )


def build_small_spec(kind: str = "single") -> TopologySpec:
    """Tiny topologies for smoke tests: ``single`` (1 switch, 2 hosts) or ``linear3``."""
    if kind == "single":
        sw = SwitchSpec(0, "s1", 1, "s1", "h1", host_ip(0), host_mac(0))
        # Second host hangs off port 2 of the same switch.
        sw.neighbor_ports = {}
        routes = {0: {host_ip(0): 1, host_ip(1): 2}}
        spec = TopologySpec("single", [sw], [], routes, [0], [[0]])
        spec._extra_hosts = [("h2", host_ip(1), host_mac(1), 0, 2)]  # type: ignore[attr-defined]
        return spec
    if kind == "linear3":
        switches = [
            SwitchSpec(i, f"s{i + 1}", i + 1, f"s{i + 1}", f"h{i + 1}", host_ip(i), host_mac(i))
            for i in range(3)
        ]
        links = [LinkSpec(0, 1, 2, 2, 2.0), LinkSpec(1, 2, 3, 2, 2.0)]
        switches[0].neighbor_ports = {1: 2}
        switches[1].neighbor_ports = {0: 2, 2: 3}
        switches[2].neighbor_ports = {1: 2}
        routes = {
            0: {host_ip(0): 1, host_ip(1): 2, host_ip(2): 2},
            1: {host_ip(0): 2, host_ip(1): 1, host_ip(2): 3},
            2: {host_ip(0): 2, host_ip(1): 2, host_ip(2): 1},
        }
        return TopologySpec("linear3", switches, links, routes, [1], [[0, 1, 2]])
    raise ValueError(f"Unknown small topology: {kind}")


class _HtbQuantumFilter:
    """Drop the harmless 'sch_htb: quantum of class ... is big' tc warning.

    Mininet echoes tc's stderr as '*** Error', once per shaped interface; the
    bandwidth limit is still applied correctly.
    """

    def filter(self, record) -> bool:
        return "sch_htb: quantum of class" not in str(record.getMessage())


def _silence_harmless_tc_warnings(logger) -> None:
    if not any(isinstance(f, _HtbQuantumFilter) for f in logger.filters):
        logger.addFilter(_HtbQuantumFilter())


def build_mininet(spec: TopologySpec, network_cfg: Dict):
    """Create (but do not start) a Mininet network for ``spec``. Requires root."""
    _paths.ensure_mininet_importable()
    from mininet.link import TCLink
    from mininet.log import lg
    from mininet.net import Mininet
    from mininet.node import OVSSwitch

    _silence_harmless_tc_warnings(lg)

    net = Mininet(switch=OVSSwitch, controller=None, link=TCLink, autoSetMacs=False, build=False)
    fail_mode = network_cfg.get("fail_mode", "secure")
    for sw in spec.switches:
        net.addSwitch(
            sw.name,
            dpid=f"{sw.dpid:016x}",
            protocols="OpenFlow13",
            failMode=fail_mode,
        )

    host_link_params = {}
    if network_cfg.get("host_link_bw_mbps"):
        host_link_params["bw"] = float(network_cfg["host_link_bw_mbps"])
    for sw in spec.switches:
        net.addHost(sw.host_name, ip=f"{sw.host_ip}/16", mac=sw.host_mac)
        net.addLink(sw.host_name, sw.name, port1=0, port2=sw.host_port, **host_link_params)
    for name, ip, mac, sw_index, port in getattr(spec, "_extra_hosts", []):
        net.addHost(name, ip=f"{ip}/16", mac=mac)
        net.addLink(name, spec.switches[sw_index].name, port1=0, port2=port, **host_link_params)

    delay_scale = float(network_cfg.get("delay_scale", 1.0))
    for link in spec.links:
        params = {}
        if network_cfg.get("link_bw_mbps"):
            params["bw"] = float(network_cfg["link_bw_mbps"])
        if network_cfg.get("use_link_delay", True):
            params["delay"] = f"{max(link.delay_ms * delay_scale, 0.01):.3f}ms"
        if network_cfg.get("max_queue_size"):
            params["max_queue_size"] = int(network_cfg["max_queue_size"])
        net.addLink(
            spec.switches[link.a].name,
            spec.switches[link.b].name,
            port1=link.port_a,
            port2=link.port_b,
            **params,
        )
    return net


def start_mininet(net) -> None:
    """Build and start the network with no controller attached yet."""
    net.build()
    net.start()
    for host in net.hosts:
        host.cmd("sysctl -qw net.ipv6.conf.all.disable_ipv6=1")
        host.cmd("sysctl -qw net.ipv6.conf.default.disable_ipv6=1")
    # Static ARP: the controllers never need to flood, so cycles are safe.
    net.staticArp()


if __name__ == "__main__":
    spec = build_topology_spec("os3e", 6)
    print(f"{spec.name}: {spec.num_switches} switches, {len(spec.links)} links, "
          f"components={[len(c) for c in spec.components]}")
    print("controller host nodes:", spec.controller_host_nodes)
    for link in spec.links[:5]:
        print(f"  s{link.a + 1}:{link.port_a} <-> s{link.b + 1}:{link.port_b}  {link.delay_ms:.2f} ms")
