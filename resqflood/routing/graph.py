"""Build a routable NetworkX graph from OSM road lines (nodes = shared vertices)."""
from __future__ import annotations

import geopandas as gpd
import networkx as nx
import numpy as np


def node_key(xy, ndigits: int = 6) -> tuple[float, float]:
    return (round(float(xy[0]), ndigits), round(float(xy[1]), ndigits))


def build_graph(roads: gpd.GeoDataFrame, metric_crs: str = "EPSG:32642") -> nx.Graph:
    """Each consecutive vertex pair of every line becomes an edge with length_m and the road's attributes.

    Expects columns produced by floodmask.ingest (flood_share, unknown_share, state); missing ones default to 0/'open'.
    """
    G = nx.Graph()
    metric = roads.to_crs(metric_crs).geometry
    for row, gm in zip(roads.itertuples(index=False), metric):
        ll = list(row.geometry.coords)
        mm = list(gm.coords)
        attrs = {
            "osm_id": getattr(row, "osm_id", None),
            "fclass": getattr(row, "fclass", None),
            "bridge": getattr(row, "bridge", "F"),
            "flood_share": float(getattr(row, "flood_share", 0.0)),
            "unknown_share": float(getattr(row, "unknown_share", 0.0)),
            "state": getattr(row, "state", "open_candidate"),
        }
        for k in range(len(ll) - 1):
            a, b = node_key(ll[k]), node_key(ll[k + 1])
            if a == b:
                continue
            seg = float(np.hypot(mm[k + 1][0] - mm[k][0], mm[k + 1][1] - mm[k][1]))
            if G.has_edge(a, b) and G[a][b]["length_m"] <= seg:
                continue
            G.add_edge(a, b, length_m=seg, **attrs)
    return G


def largest_component(G: nx.Graph) -> nx.Graph:
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()


def nearest_node(G: nx.Graph, lon: float, lat: float) -> tuple[tuple[float, float], float]:
    """Return (node, approx distance in metres)."""
    nodes = np.array(list(G.nodes))
    d = np.hypot((nodes[:, 0] - lon) * 111320 * np.cos(np.radians(lat)), (nodes[:, 1] - lat) * 110574)
    i = int(np.argmin(d))
    return tuple(nodes[i]), float(d[i])
