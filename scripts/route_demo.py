"""Flood-aware routes between two points (default Johi -> Dadu city).
python scripts/route_demo.py --date 20220830_D078 --k 3
"""
import argparse
import json

import _common  # noqa: F401
import geopandas as gpd
from shapely.geometry import LineString

from resqflood.config import load_config
from resqflood.floodmask.ingest import ingest_mask
from resqflood.routing.graph import build_graph, largest_component
from resqflood.routing.router import apply_costs, route, shortest_by_length

ap = argparse.ArgumentParser()
ap.add_argument("--date", default=None)
ap.add_argument("--start", default="67.6146,26.6927", help="lon,lat (default Johi)")
ap.add_argument("--end", default="67.7797,26.7311", help="lon,lat (default Dadu city)")
ap.add_argument("--k", type=int, default=3)
a = ap.parse_args()
cfg = load_config()
key = a.date or cfg["flood"]["default_date"]
s = tuple(map(float, a.start.split(","))); t = tuple(map(float, a.end.split(",")))
roads = ingest_mask(gpd.read_file(cfg.data(cfg["roads"]["file"])), cfg.flood_mask(key), cfg)
G = largest_component(build_graph(roads, cfg["study_area"]["metric_crs"]))
apply_costs(G, **cfg["roads"]["cost"])
res = route(G, s, t, k=a.k)
short = shortest_by_length(G, s, t)
print(f"graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges | snaps {res['start_snap_m']} m / {res['end_snap_m']} m")
print(f"shortest by distance: {short['length_km']} km, {short['flood_exposed_km']} km flooded")
for i, r in enumerate(res["routes"], 1):
    print(f"flood-aware route {i}: {r['length_km']} km, {r['flood_exposed_km']} km flooded")
rows = [("shortest_by_distance", short)] + [(f"flood_aware_{i}", r) for i, r in enumerate(res["routes"], 1)]
gdf = gpd.GeoDataFrame({"route": [n for n, _ in rows], "length_km": [r["length_km"] for _, r in rows],
                        "flood_exposed_km": [r["flood_exposed_km"] for _, r in rows]},
                       geometry=[LineString(r["coords"]) for _, r in rows], crs=4326)
out = cfg.work_dir / f"routes_{key}.geojson"; gdf.to_file(out, driver="GeoJSON"); print("wrote", out)
