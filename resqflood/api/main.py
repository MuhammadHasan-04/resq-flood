"""FastAPI service. Run from the repo root:  uvicorn resqflood.api.main:app --reload"""
from __future__ import annotations

from functools import lru_cache

import geopandas as gpd
from fastapi import FastAPI, HTTPException

from resqflood.config import load_config
from resqflood.floodmask.ingest import ingest_mask, summary
from resqflood.reports.schema import Report, ReportIn
from resqflood.reports.validate import ReportValidator, verified_statuses
from resqflood.routing.graph import build_graph, largest_component
from resqflood.routing.router import apply_costs, route, shortest_by_length

app = FastAPI(title="ResQFlood API", version="0.1.0")
cfg = load_config()
REPORTS: list[Report] = []


@lru_cache(maxsize=4)
def scored_roads(date_key: str):
    roads = gpd.read_file(cfg.data(cfg["roads"]["file"]))
    return ingest_mask(roads, cfg.flood_mask(date_key), cfg)


@lru_cache(maxsize=4)
def graph(date_key: str):
    return largest_component(build_graph(scored_roads(date_key), cfg["study_area"]["metric_crs"]))


@lru_cache(maxsize=1)
def validator():
    sector = gpd.read_file(cfg.data(cfg["study_area"]["pilot_sector"])).geometry.iloc[0]
    sr = scored_roads(cfg["flood"]["default_date"])
    return ReportValidator(cfg, sr, sector, dict(zip(sr.osm_id, sr.state)))


@app.get("/health")
def health():
    return {"status": "ok", "data_root": str(cfg.data_root), "data_found": cfg.data_root.exists()}


@app.get("/flood/dates")
def dates():
    return cfg["flood"]["dates"]


@app.get("/flood/{date_key}/summary")
def flood_summary(date_key: str):
    if date_key not in cfg["flood"]["dates"]:
        raise HTTPException(404, f"unknown date {date_key}")
    return summary(scored_roads(date_key))


@app.get("/route")
def get_route(start_lon: float, start_lat: float, end_lon: float, end_lat: float,
              date_key: str | None = None, k: int = 3):
    key = date_key or cfg["flood"]["default_date"]
    G = graph(key)
    apply_costs(G, cfg["roads"]["cost"]["flood_weight"], cfg["roads"]["cost"]["unknown_weight"], verified_statuses(REPORTS))
    res = route(G, (start_lon, start_lat), (end_lon, end_lat), k=k)
    res["shortest_by_distance"] = shortest_by_length(G, (start_lon, start_lat), (end_lon, end_lat))
    res["mask_date"] = key
    return res


@app.post("/reports", response_model=Report)
def post_report(r: ReportIn):
    rep = validator().check(r)
    REPORTS.append(rep)
    return rep


@app.get("/reports", response_model=list[Report])
def list_reports():
    return REPORTS
