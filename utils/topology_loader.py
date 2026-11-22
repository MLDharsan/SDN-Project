#!/usr/bin/env python3
"""
Real topology loader for Topology Zoo `.gml` files used in experiments.

This loader:
1. Parses the real graph structure from disk.
2. Recovers node coordinates from GML attributes or city-level fallbacks.
3. Computes edge delays from geographic distance.
4. Builds shortest-path delay matrices for controller-placement experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import math

import networkx as nx
import numpy as np


FIBER_KM_PER_MS = 200.0


@dataclass
class LoadedTopology:
    name: str
    graph: nx.Graph
    ordered_nodes: List[int]
    node_labels: Dict[int, str]
    coordinates: Dict[int, Optional[Tuple[float, float]]]
    edge_delay_ms: Dict[Tuple[int, int], float]
    shortest_path_delay_ms: np.ndarray


class TopologyLoader:
    """Load real network topologies and compute geographic shortest-path delays."""

    FILE_MAP = {
        'gridnet': 'Gridnet.gml',
        'bellcanada': 'BellCanada.gml',
        'os3e': 'Os3e.gml',
        'interoute': 'Interoute.gml',
        'cogentco': 'Cogentco.gml',
    }

    # OS3E lacks coordinates in the provided GML, so we use city-center fallbacks.
    CITY_COORDS = {
        'Amsterdam': (52.3676, 4.9041),
        'Athens': (37.9838, 23.7275),
        'Belgrade': (44.7866, 20.4489),
        'Berlin': (52.5200, 13.4050),
        'Bern': (46.9480, 7.4474),
        'Bratislava': (48.1486, 17.1077),
        'Brussels': (50.8503, 4.3517),
        'Bucharest': (44.4268, 26.1025),
        'Budapest': (47.4979, 19.0402),
        'Copenhagen': (55.6761, 12.5683),
        'Dublin': (53.3498, -6.2603),
        'Geneva': (46.2044, 6.1432),
        'Hamburg': (53.5511, 9.9937),
        'Helsinki': (60.1699, 24.9384),
        'Istanbul': (41.0082, 28.9784),
        'Kiev': (50.4501, 30.5234),
        'Kyiv': (50.4501, 30.5234),
        'Lisbon': (38.7223, -9.1393),
        'Ljubljana': (46.0569, 14.5058),
        'London': (51.5072, -0.1276),
        'Luxembourg': (49.6116, 6.1319),
        'Lyon': (45.7640, 4.8357),
        'Madrid': (40.4168, -3.7038),
        'Milan': (45.4642, 9.1900),
        'Moscow': (55.7558, 37.6173),
        'Munich': (48.1351, 11.5820),
        'Oslo': (59.9139, 10.7522),
        'Paris': (48.8566, 2.3522),
        'Prague': (50.0755, 14.4378),
        'Reykjavik': (64.1466, -21.9426),
        'Riga': (56.9496, 24.1052),
        'Rome': (41.9028, 12.4964),
        'Stockholm': (59.3293, 18.0686),
        'Vienna': (48.2082, 16.3738),
        'Warsaw': (52.2297, 21.0122),
    }

    def __init__(self, base_dir: Optional[str] = None):
        self.base_dir = Path(base_dir or Path(__file__).resolve().parents[1] / 'data' / 'topologies')

    @staticmethod
    def normalize_name(topology_name: str) -> str:
        return (topology_name or 'os3e').strip().lower()

    def get_topology_path(self, topology_name: str) -> Path:
        key = self.normalize_name(topology_name)
        filename = self.FILE_MAP.get(key)
        if filename is None:
            raise FileNotFoundError(f"Unsupported topology: {topology_name}")
        path = self.base_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Topology file not found: {path}")
        return path

    def load_topology(self, topology_name: str) -> LoadedTopology:
        key = self.normalize_name(topology_name)
        path = self.get_topology_path(key)

        try:
            graph = nx.read_gml(path, label='id')
        except nx.NetworkXError as exc:
            if 'duplicated' not in str(exc):
                raise
            lines = path.read_text().splitlines()
            if len(lines) > 1 and 'multigraph' not in lines[1]:
                lines.insert(1, '  multigraph 1')
            graph = nx.parse_gml(lines, label='id')

        graph = nx.Graph(graph)
        graph.remove_edges_from(nx.selfloop_edges(graph))

        ordered_nodes = sorted(graph.nodes())
        label_map = {
            node: str(graph.nodes[node].get('label') or graph.nodes[node].get('name') or node)
            for node in ordered_nodes
        }
        coords = {
            node: self._extract_coordinates(label_map[node], graph.nodes[node])
            for node in ordered_nodes
        }

        edge_delay_ms = {}
        for u, v in graph.edges():
            delay_ms = self._edge_delay_ms(coords[u], coords[v])
            graph[u][v]['delay_ms'] = delay_ms
            edge_delay_ms[(u, v)] = delay_ms
            edge_delay_ms[(v, u)] = delay_ms

        shortest_paths = dict(nx.all_pairs_dijkstra_path_length(graph, weight='delay_ms'))
        delay_matrix = np.zeros((len(ordered_nodes), len(ordered_nodes)), dtype=np.float32)
        node_index = {node: idx for idx, node in enumerate(ordered_nodes)}
        for source in ordered_nodes:
            source_idx = node_index[source]
            lengths = shortest_paths.get(source, {})
            for target in ordered_nodes:
                target_idx = node_index[target]
                delay_matrix[source_idx, target_idx] = float(lengths.get(target, 0.0))

        return LoadedTopology(
            name=key,
            graph=graph,
            ordered_nodes=ordered_nodes,
            node_labels=label_map,
            coordinates=coords,
            edge_delay_ms=edge_delay_ms,
            shortest_path_delay_ms=delay_matrix,
        )

    def get_topology_info(self, topology_name: str) -> Dict:
        topo = self.load_topology(topology_name)
        return {
            'name': topo.name,
            'num_nodes': topo.graph.number_of_nodes(),
            'num_edges': topo.graph.number_of_edges(),
            'node_labels': topo.node_labels,
            'coordinates': topo.coordinates,
            'graph': topo.graph,
            'shortest_path_delay_ms': topo.shortest_path_delay_ms,
        }

    def build_switch_to_controller_delay_matrix(
        self,
        topology_name: str,
        total_controllers: int,
    ) -> Tuple[LoadedTopology, np.ndarray, List[int]]:
        topo = self.load_topology(topology_name)
        controller_hosts = self._select_controller_hosts(topo.graph, topo.ordered_nodes, total_controllers)
        node_index = {node: idx for idx, node in enumerate(topo.ordered_nodes)}

        delay_matrix = np.zeros((len(topo.ordered_nodes), total_controllers), dtype=np.float32)
        for switch_idx, switch_node in enumerate(topo.ordered_nodes):
            source_idx = node_index[switch_node]
            for controller_idx, host_node in enumerate(controller_hosts):
                host_idx = node_index[host_node]
                delay_matrix[switch_idx, controller_idx] = topo.shortest_path_delay_ms[source_idx, host_idx]

        return topo, delay_matrix, controller_hosts

    @staticmethod
    def _extract_coordinates(label: str, attrs: Dict) -> Optional[Tuple[float, float]]:
        lat = attrs.get('Latitude')
        lon = attrs.get('Longitude')
        if lat is not None and lon is not None:
            return float(lat), float(lon)
        return TopologyLoader.CITY_COORDS.get(label)

    @staticmethod
    def _edge_delay_ms(
        coord_u: Optional[Tuple[float, float]],
        coord_v: Optional[Tuple[float, float]],
    ) -> float:
        if coord_u is None or coord_v is None:
            return 1.0

        distance_km = TopologyLoader._haversine_km(coord_u, coord_v)
        return max(distance_km / FIBER_KM_PER_MS, 0.1)

    @staticmethod
    def _haversine_km(coord_u: Tuple[float, float], coord_v: Tuple[float, float]) -> float:
        lat1, lon1 = coord_u
        lat2, lon2 = coord_v
        radius_km = 6371.0

        d_lat = math.radians(lat2 - lat1)
        d_lon = math.radians(lon2 - lon1)
        a = (
            math.sin(d_lat / 2) ** 2
            + math.cos(math.radians(lat1))
            * math.cos(math.radians(lat2))
            * math.sin(d_lon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return radius_km * c

    @staticmethod
    def _select_controller_hosts(
        graph: nx.Graph,
        ordered_nodes: List[int],
        total_controllers: int,
    ) -> List[int]:
        closeness = nx.closeness_centrality(graph)
        ranked_nodes = sorted(
            ordered_nodes,
            key=lambda node: (
                -graph.degree[node],
                -closeness[node],
                node,
            ),
        )
        selected = ranked_nodes[:total_controllers]
        if len(selected) < total_controllers:
            selected.extend(ordered_nodes[: total_controllers - len(selected)])
        return selected
