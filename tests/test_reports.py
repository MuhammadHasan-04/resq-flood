from datetime import datetime, timezone

import geopandas as gpd
from shapely.geometry import LineString, box

from resqflood.reports.schema import ReportIn, ReviewState
from resqflood.reports.validate import ReportValidator, verified_statuses

CFG = {"reports": {"snap_distance_m": 50, "duplicate_window_h": 2, "duplicate_radius_m": 200,
                   "role_weights": {"public": 0.4, "ngo_worker": 0.7, "field_team": 0.9, "authority": 1.0}, "accept_score": 0.6},
       "study_area": {"metric_crs": "EPSG:32642"}}
ROADS = gpd.GeoDataFrame({"osm_id": [10]}, geometry=[LineString([(67.60, 26.70), (67.62, 26.70)])], crs=4326)
SECTOR = box(67.35, 26.45, 68.05, 27.05)
T = datetime(2022, 8, 30, 8, tzinfo=timezone.utc)


def rep(lon=67.61, lat=26.70005, role="authority", kind="road_blocked"):
    return ReportIn(lon=lon, lat=lat, timestamp_utc=T, report_type=kind, source_role=role)


def test_outside_sector_rejected():
    v = ReportValidator(CFG, ROADS, SECTOR)
    assert v.check(rep(lon=66.0)).review_state == ReviewState.rejected


def test_far_from_road_rejected():
    v = ReportValidator(CFG, ROADS, SECTOR)
    assert v.check(rep(lat=26.71)).review_state == ReviewState.rejected


def test_authority_report_agreeing_with_map_accepted():
    v = ReportValidator(CFG, ROADS, SECTOR, {10: "blocked_candidate"})
    r = v.check(rep())
    assert r.review_state == ReviewState.accepted and r.osm_road_id == 10
    assert verified_statuses([r]) == {10: "blocked"}


def test_single_public_report_stays_pending():
    v = ReportValidator(CFG, ROADS, SECTOR)
    assert v.check(rep(role="public")).review_state == ReviewState.pending


def test_conflicting_reports_flagged():
    v = ReportValidator(CFG, ROADS, SECTOR)
    v.check(rep(kind="road_open"))
    assert v.check(rep(kind="road_blocked")).review_state == ReviewState.conflict
