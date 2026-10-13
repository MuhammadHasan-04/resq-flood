"""Flood-mask ingestion: score every road segment against a dated TU Wien (or model) flood mask.

Mask values: 0 = observed, not flooded; 1 = flood; 255 = not observable (masked).
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import rasterio

STATE_ORDER = ["blocked_candidate", "risky", "unknown", "open_candidate"]


def sample_line_shares(lines_ll: gpd.GeoSeries, lines_m: gpd.GeoSeries, mask_path, step_m: float = 20.0,
                       flood_value: int = 1, nodata: int = 255) -> tuple[np.ndarray, np.ndarray]:
    """Return (flood_share, unknown_share) per line, sampling the mask every `step_m` metres."""
    with rasterio.open(mask_path) as src:
        arr = src.read(1)
        h, w = arr.shape
        flood, unknown = np.zeros(len(lines_ll)), np.zeros(len(lines_ll))
        for i, (g_ll, g_m) in enumerate(zip(lines_ll, lines_m)):
            n = max(2, int(g_m.length / step_m))
            pts = [g_ll.interpolate(t, normalized=True) for t in np.linspace(0, 1, n)]
            rows, cols = rasterio.transform.rowcol(src.transform, [p.x for p in pts], [p.y for p in pts])
            rows, cols = np.asarray(rows), np.asarray(cols)
            inside = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
            vals = np.full(n, nodata)
            vals[inside] = arr[rows[inside], cols[inside]]
            flood[i] = (vals == flood_value).mean()
            unknown[i] = (vals == nodata).mean()
    return flood, unknown


def classify(flood_share: float, unknown_share: float, rules: dict) -> str:
    if flood_share >= rules["blocked_candidate"]["flood_share_min"]:
        return "blocked_candidate"
    if flood_share >= rules["risky"]["flood_share_min"]:
        return "risky"
    if unknown_share >= rules["unknown"]["unknown_share_min"]:
        return "unknown"
    return "open_candidate"


def ingest_mask(roads: gpd.GeoDataFrame, mask_path, cfg) -> gpd.GeoDataFrame:
    """Return a copy of `roads` with flood_share, unknown_share, state, length_m and mask_date columns."""
    rc = cfg["roads"]
    vals = cfg["flood"]["mask_values"]
    out = roads.copy()
    metric = out.to_crs(cfg["study_area"]["metric_crs"]).geometry
    f, u = sample_line_shares(out.geometry, metric, mask_path, rc["sample_step_m"], vals["flood"], vals["nodata"])
    out["length_m"] = metric.length.round(1)
    out["flood_share"] = f.round(4)
    out["unknown_share"] = u.round(4)
    out["state"] = [classify(a, b, rc["state_rules"]) for a, b in zip(f, u)]
    out["mask_file"] = str(mask_path).split("/")[-1].split("\\")[-1]
    return out


def summary(scored: gpd.GeoDataFrame) -> dict:
    return {
        "roads": int(len(scored)),
        "road_km": round(scored.length_m.sum() / 1000, 1),
        "flood_exposed_km": round((scored.flood_share * scored.length_m).sum() / 1000, 1),
        "states": {s: int((scored.state == s).sum()) for s in STATE_ORDER},
    }
