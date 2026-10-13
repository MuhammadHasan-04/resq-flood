"""Flood-aware road routing: A* (default), Dijkstra, and Yen's k-shortest alternatives."""
from __future__ import annotations

from itertools import islice

import networkx as nx
import numpy as np

from .graph import nearest_node


def apply_costs(G: nx.Graph, flood_weight: float = 20.0, unknown_weight: float = 2.0,
                verified: dict | None = None) -> None:
    """Set edge 'cost' = length * (1 + fw*flood + uw*unknown).

    `verified` maps osm_id -> 'blocked' | 'open' from accepted crowd reports:
    blocked edges get cost=inf (removed from routing), open edges drop their flood penalty.
    """
    verified = verified or {}
    for _, _, d in G.edges(data=True):
        status = verified.get(d.get("osm_id"))
        if status == "blocked":
            d["cost"] = float("inf")
        elif status == "open":
            d["cost"] = d["length_m"]
        else:
            d["cost"] = d["length_m"] * (1 + flood_weight * d["flood_share"] + unknown_weight * d["unknown_share"])


def _heuristic(u, v) -> float:
    # straight-line metres; admissible because cost >= length
    return float(np.hypot((u[0] - v[0]) * 111320 * np.cos(np.radians(u[1])), (u[1] - v[1]) * 110574))


def _usable(G: nx.Graph) -> nx.Graph:
    return nx.subgraph_view(G, filter_edge=lambda a, b: np.isfinite(G[a][b]["cost"]))


def alternatives_by_penalty(H: nx.Graph, s, t, k: int = 3, factor: float = 1.6, max_tries: int = 15) -> list:
    """A* repeatedly; after each route multiply the cost of its edges by `factor` so the next one diverges."""
    extra: dict = {}
    w = lambda a, b, d: d["cost"] * extra.get(frozenset((a, b)), 1.0)
    paths: list = []
    for _ in range(max_tries):
        p = nx.astar_path(H, s, t, heuristic=_heuristic, weight=w)
        if p not in paths:
            paths.append(p)
        if len(paths) >= k:
            break
        for a, b in zip(p, p[1:]):
            key = frozenset((a, b))
            extra[key] = extra.get(key, 1.0) * factor
    return paths


def path_stats(G: nx.Graph, path: list) -> dict:
    edges = list(zip(path, path[1:]))
    length = sum(G[a][b]["length_m"] for a, b in edges)
    flooded = sum(G[a][b]["length_m"] * G[a][b]["flood_share"] for a, b in edges)
    return {"length_km": round(length / 1000, 2), "flood_exposed_km": round(flooded / 1000, 2),
            "cost": round(sum(G[a][b]["cost"] for a, b in edges), 1), "edges": len(edges),
            "coords": [list(p) for p in path]}


def route(G: nx.Graph, start_lonlat, end_lonlat, method: str = "astar", k: int = 1) -> dict:
    """Return start/end snap distances and up to k routes ranked by cost."""
    s, ds = nearest_node(G, *start_lonlat)
    t, dt = nearest_node(G, *end_lonlat)
    H = _usable(G)
    if method == "dijkstra":
        paths = [nx.dijkstra_path(H, s, t, weight="cost")]
    elif method == "yen":  # exact k-shortest simple paths; slow on big graphs, alternatives often near-identical
        paths = list(islice(nx.shortest_simple_paths(H, s, t, weight="cost"), k))
    else:  # A*; k>1 uses the penalty method to get genuinely different corridors
        paths = alternatives_by_penalty(H, s, t, k)
    return {"start_snap_m": round(ds, 1), "end_snap_m": round(dt, 1), "routes": [path_stats(G, p) for p in paths]}


def shortest_by_length(G: nx.Graph, start_lonlat, end_lonlat) -> dict:
    """Plain shortest-distance route, for comparison with the flood-aware one."""
    s, _ = nearest_node(G, *start_lonlat)
    t, _ = nearest_node(G, *end_lonlat)
    return path_stats(G, nx.astar_path(G, s, t, heuristic=_heuristic, weight="length_m"))
