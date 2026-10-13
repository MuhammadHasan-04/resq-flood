import numpy as np
import rasterio
import geopandas as gpd
from affine import Affine
from shapely.geometry import LineString

from resqflood.floodmask.ingest import classify, ingest_mask

RULES = {"blocked_candidate": {"flood_share_min": 0.5}, "risky": {"flood_share_min": 1e-6}, "unknown": {"unknown_share_min": 0.5}}


def test_classify():
    assert classify(0.6, 0, RULES) == "blocked_candidate"
    assert classify(0.1, 0, RULES) == "risky"
    assert classify(0, 0.8, RULES) == "unknown"
    assert classify(0, 0, RULES) == "open_candidate"


def test_ingest_on_synthetic_mask(tmp_path):
    arr = np.zeros((100, 100), np.uint8); arr[:, 50:] = 1          # east half flooded
    T = Affine(0.001, 0, 67.0, 0, -0.001, 27.0)
    p = tmp_path / "m.tif"
    with rasterio.open(p, "w", driver="GTiff", height=100, width=100, count=1, dtype="uint8", crs="EPSG:4326", transform=T, nodata=255) as o:
        o.write(arr, 1)
    roads = gpd.GeoDataFrame({"osm_id": [1, 2]}, geometry=[LineString([(67.005, 26.95), (67.045, 26.95)]),
                                                           LineString([(67.055, 26.95), (67.095, 26.95)])], crs=4326)
    cfg = {"roads": {"sample_step_m": 20, "state_rules": RULES}, "flood": {"mask_values": {"flood": 1, "nodata": 255}},
           "study_area": {"metric_crs": "EPSG:32642"}}
    out = ingest_mask(roads, p, cfg)
    assert list(out.state) == ["open_candidate", "blocked_candidate"]
