"""Rule-based validation and credibility scoring for crowd reports.

Score components follow the VGI credibility idea of Safariallahkheili & Malek (2022): location plausibility,
agreement with nearby reports, agreement with the flood map, plus a source-role weight.
Only reports with score >= accept_score become 'accepted'; nothing is ever auto-accepted without these checks.
"""
from __future__ import annotations

from datetime import timedelta

import geopandas as gpd
import numpy as np
from shapely.geometry import Point

from .schema import Report, ReportIn, ReviewState


class ReportValidator:
    def __init__(self, cfg, roads: gpd.GeoDataFrame, sector_geom, flood_state_by_osm: dict | None = None):
        self.cfg = cfg["reports"]
        self.crs = cfg["study_area"]["metric_crs"]
        self.roads_m = roads.to_crs(self.crs)
        self.sindex = self.roads_m.sindex
        self.sector_m = gpd.GeoSeries([sector_geom], crs=4326).to_crs(self.crs).iloc[0]
        self.flood_state = flood_state_by_osm or {}
        self.history: list[Report] = []

    def _point_m(self, lon, lat):
        return gpd.GeoSeries([Point(lon, lat)], crs=4326).to_crs(self.crs).iloc[0]

    def snap(self, lon, lat):
        p = self._point_m(lon, lat)
        idx = list(self.sindex.nearest(p, return_all=False)[1])[0]
        row = self.roads_m.iloc[idx]
        return int(row["osm_id"]), float(row.geometry.distance(p)), p

    def check(self, rep_in: ReportIn) -> Report:
        rep = Report(**rep_in.model_dump())
        osm_id, dist, p = self.snap(rep.lon, rep.lat)
        rep.osm_road_id, rep.snap_distance_m = osm_id, round(dist, 1)
        reasons, score = [], 0.0

        if not self.sector_m.contains(p):
            rep.review_state, rep.reasons = ReviewState.rejected, ["outside pilot sector"]
            return rep
        if dist > self.cfg["snap_distance_m"]:
            rep.review_state, rep.reasons = ReviewState.rejected, [f"no road within {self.cfg['snap_distance_m']} m"]
            return rep

        # duplicates: same road and type within the time window
        win = timedelta(hours=self.cfg["duplicate_window_h"])
        same = [h for h in self.history if h.osm_road_id == osm_id and h.report_type == rep.report_type
                and abs(h.timestamp_utc - rep.timestamp_utc) <= win]
        opposite = [h for h in self.history if h.osm_road_id == osm_id and abs(h.timestamp_utc - rep.timestamp_utc) <= win
                    and {h.report_type.value, rep.report_type.value} == {"road_open", "road_blocked"}]

        role_w = self.cfg["role_weights"][rep.source_role.value]
        score += 0.4 * role_w;                       reasons.append(f"role {rep.source_role.value} (+{0.4 * role_w:.2f})")
        score += 0.2 * (1 - dist / self.cfg["snap_distance_m"]); reasons.append(f"snapped {dist:.0f} m")
        if same:
            score += min(0.3, 0.15 * len(same));     reasons.append(f"{len(same)} matching nearby report(s)")
        state = self.flood_state.get(osm_id)
        if state is not None:
            agrees = (rep.report_type.value == "road_blocked" and state in ("blocked_candidate", "risky")) or \
                     (rep.report_type.value == "road_open" and state == "open_candidate")
            if agrees:
                score += 0.2;                         reasons.append(f"agrees with flood map ({state})")
            else:
                reasons.append(f"differs from flood map ({state})")
        rep.score = round(min(score, 1.0), 3)
        if opposite:
            rep.review_state = ReviewState.conflict;  reasons.append("conflicts with a recent opposite report")
        elif rep.score >= self.cfg["accept_score"]:
            rep.review_state = ReviewState.accepted
        rep.reasons = reasons
        self.history.append(rep)
        return rep


def verified_statuses(reports: list[Report]) -> dict:
    """osm_id -> 'blocked' | 'open' from accepted reports (latest wins), for routing.apply_costs."""
    out = {}
    for r in sorted((r for r in reports if r.review_state == ReviewState.accepted), key=lambda r: r.timestamp_utc):
        if r.report_type.value in ("road_blocked", "bridge_damaged"):
            out[r.osm_road_id] = "blocked"
        elif r.report_type.value == "road_open":
            out[r.osm_road_id] = "open"
    return out
