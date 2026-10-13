"""Raster cost grid and least-cost boat path for the pilot sector.

cell_cost = distance + obstacle + uncertainty + terrain + report
"""
from __future__ import annotations

import geopandas as gpd
import numpy as np
import rasterio
from affine import Affine
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject
from scipy.ndimage import label
from skimage.graph import route_through_array


def grid(cfg):
    x0, y0, x1, y1 = cfg["study_area"]["pilot_bbox"]
    r = cfg["boat"]["cell_deg"]
    W, H = int(round((x1 - x0) / r)), int(round((y1 - y0) / r))
    return Affine(r, 0, x0, 0, -r, y1), (H, W)


def _warp(path, T, shape, resampling, fill, dtype):
    out = np.full(shape, fill, dtype)
    with rasterio.open(path) as s:
        reproject(s.read(1), out, src_transform=s.transform, src_crs=s.crs, dst_transform=T, dst_crs="EPSG:4326",
                  resampling=resampling, src_nodata=s.nodata, dst_nodata=fill)
    return out


def build_cost_grid(cfg, mask_path, roads: gpd.GeoDataFrame, report_penalty: np.ndarray | None = None) -> dict:
    T, shape = grid(cfg)
    b, pen = cfg["boat"], cfg["boat"]["penalties"]
    fl = _warp(mask_path, T, shape, Resampling.mode, 255, np.uint8)
    perm = _warp(cfg.data(cfg["flood"]["permanent_water"]), T, shape, Resampling.mode, 255, np.uint8)
    occ = _warp(cfg.data(b["jrc_occurrence"]), T, shape, Resampling.average, 255, np.uint8)
    dem = _warp(cfg.data(b["dem"]), T, shape, Resampling.average, -9999, np.float32)
    slope = _warp(cfg.data(b["slope"]), T, shape, Resampling.average, -9999, np.float32)
    water = (fl == 1) | (perm == 1) | ((occ >= 50) & (occ <= 100))
    nb = roads[roads.get("bridge", "F") != "T"]
    br = roads[roads.get("bridge", "F") == "T"]
    road_r = rasterize(((g, 1) for g in nb.geometry), out_shape=shape, transform=T, all_touched=True) if len(nb) else np.zeros(shape, np.uint8)
    br_r = rasterize(((g, 1) for g in br.geometry), out_shape=shape, transform=T, all_touched=True) if len(br) else np.zeros(shape, np.uint8)
    valid_dem = dem > -9999
    wl = float(np.percentile(dem[(fl == 1) & valid_dem], b["water_level_percentile"])) if ((fl == 1) & valid_dem).any() else 0.0
    cost = (1.0
            + np.where(water, 0, pen["dry"]) + pen["road"] * road_r + pen["bridge"] * br_r
            + np.where(fl == 255, pen["masked"], 0) + np.where(valid_dem, 0, pen["dem_gap"])
            + np.where(valid_dem, np.clip(dem - wl, 0, pen["max_terrain"]), pen["dem_gap"])
            + np.where(slope > 2, pen["slope_gt2"], 0))
    if report_penalty is not None:
        cost = cost + report_penalty
    return {"cost": cost.astype(np.float32), "water": water, "transform": T, "flood_level_m": wl, "road": road_r}


def to_rc(T: Affine, lon: float, lat: float):
    c, r = ~T * (lon, lat)
    return int(r), int(c)


def nearest_big_water(water: np.ndarray, rc, min_cells: int = 20000):
    lab, _ = label(water, structure=np.ones((3, 3)))
    sizes = np.bincount(lab.ravel())
    keep = np.where(sizes >= min_cells)[0]
    keep = keep[keep != 0]
    big = np.isin(lab, keep)
    rr, cc = np.where(big)
    if rr.size == 0:
        rr, cc = np.where(water)
    i = int(np.argmin((rr - rc[0]) ** 2 + (cc - rc[1]) ** 2))
    return int(rr[i]), int(cc[i])


def boat_path(g: dict, start_lonlat, end_lonlat) -> dict:
    T, water, cost = g["transform"], g["water"], g["cost"]
    s = nearest_big_water(water, to_rc(T, *start_lonlat))
    t = nearest_big_water(water, to_rc(T, *end_lonlat))
    path, total = route_through_array(cost, s, t, fully_connected=True, geometric=True)
    p = np.array(path)
    r = T.a
    km = float(np.sum(np.hypot(np.diff(p[:, 0]) * r * 110574, np.diff(p[:, 1]) * r * 99500))) / 1000
    coords = [list(T * (c + 0.5, rr + 0.5)) for rr, c in path]
    return {"length_km": round(km, 2), "cells": len(path), "cost": round(float(total), 1),
            "dry_cells": int((~water[p[:, 0], p[:, 1]]).sum()),
            "road_crossings": int(g["road"][p[:, 0], p[:, 1]].sum()), "coords": coords}
