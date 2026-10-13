"""Synthetic road network: a short flooded road vs a longer dry detour."""
import geopandas as gpd
from shapely.geometry import LineString

from resqflood.routing.graph import build_graph
from resqflood.routing.router import apply_costs, route, shortest_by_length

A, B, C = (67.60, 26.70), (67.62, 26.70), (67.61, 26.71)


def net():
    return gpd.GeoDataFrame({"osm_id": [1, 2, 3], "fclass": ["primary"] * 3, "bridge": ["F"] * 3,
                             "flood_share": [0.9, 0.0, 0.0], "unknown_share": [0.0] * 3,
                             "state": ["blocked_candidate", "open_candidate", "open_candidate"]},
                            geometry=[LineString([A, B]), LineString([A, C]), LineString([C, B])], crs=4326)


def test_flood_aware_avoids_flooded_road():
    G = build_graph(net()); apply_costs(G, 20, 2)
    r = route(G, A, B, k=1)["routes"][0]
    assert r["flood_exposed_km"] == 0 and r["edges"] == 2


def test_shortest_takes_flooded_road():
    G = build_graph(net()); apply_costs(G, 20, 2)
    assert shortest_by_length(G, A, B)["edges"] == 1


def test_verified_open_removes_penalty():
    G = build_graph(net()); apply_costs(G, 20, 2, verified={1: "open"})
    assert route(G, A, B)["routes"][0]["edges"] == 1


def test_verified_blocked_removes_edge():
    G = build_graph(net()); apply_costs(G, 0, 0, verified={1: "blocked"})
    assert route(G, A, B)["routes"][0]["edges"] == 2


def test_k_alternatives_distinct():
    G = build_graph(net()); apply_costs(G, 20, 2)
    rs = route(G, A, B, k=2)["routes"]
    assert len(rs) == 2 and rs[0]["coords"] != rs[1]["coords"]
