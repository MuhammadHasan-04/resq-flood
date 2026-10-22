import numpy as np
import pytest

from resqflood.detect import DEFAULT_PARAMS, decide
from resqflood.sar_io import InputError, parse_location


def test_parse_location():
    assert parse_location("67.6 26.6 67.8 26.8") == ("bbox", [67.6, 26.6, 67.8, 26.8])
    assert parse_location("26.7, 67.7")[0] == "center"
    assert parse_location("") is None
    with pytest.raises(InputError):
        parse_location("67.8 26.6 67.6 26.8")       # west > east


def test_river_only_scene_gives_no_flood():
    valid = np.ones((200, 200), bool)
    perm = np.zeros_like(valid)
    perm[:, 90:110] = True                           # river
    prob = np.zeros(valid.shape, "float32")
    prob[perm] = 0.99
    prob[:, 88:90] = 0.9                             # wet bank right next to the river
    fb = prob > 0.5
    p = dict(DEFAULT_PARAMS, t_high=0.5, t_low=0.5, verdict_buffer_px=2, min_blob_px=10, min_frac=0.005, min_km2=1.0)
    d = decide(prob, fb, valid, perm, np.zeros_like(valid), p, px_km2=0.0004)
    assert not d["verdict"]
    assert (d["classes"][:, 95] == 3).all()          # river shown as permanent water


def test_flood_next_to_river_is_detected():
    valid = np.ones((200, 200), bool)
    perm = np.zeros_like(valid)
    perm[:, 90:110] = True
    prob = np.zeros(valid.shape, "float32")
    prob[perm] = 0.99
    prob[20:120, 120:190] = 0.95                     # flooded fields
    fb = prob > 0.5
    p = dict(DEFAULT_PARAMS, t_high=0.5, t_low=0.5, verdict_buffer_px=2, min_blob_px=10, min_frac=0.005, min_km2=1.0)
    d = decide(prob, fb, valid, perm, np.zeros_like(valid), p, px_km2=0.0004)
    assert d["verdict"]
    assert (d["classes"][50, 150] == 2) and (d["classes"][50, 100] == 3)


def test_fallback_only_when_no_model():
    valid = np.ones((100, 100), bool)
    fb = np.zeros_like(valid)
    fb[:50] = True
    d = decide(None, fb, valid, np.zeros_like(valid), np.zeros_like(valid), dict(DEFAULT_PARAMS), px_km2=0.0004)
    assert d["verdict"] and (d["classes"][:50] == 2).all()


def _write_safe(tmp_path, sigma0_vv, sigma0_vh, A=400.0):
    import zipfile
    import rasterio
    from rasterio.control import GroundControlPoint
    H, W = sigma0_vv.shape
    g = [GroundControlPoint(row=i, col=j, x=67.0 + j * 0.0002, y=27.0 - (H - i) * 0.0002, z=0)
         for i in (0, H) for j in (0, W)] + [GroundControlPoint(row=H / 2, col=W / 2, x=67.0 + W * 0.0001, y=27.0 - H * 0.0001, z=0)]
    zp = tmp_path / "S1A_IW_GRDH_TEST.zip"
    with zipfile.ZipFile(zp, "w") as z:
        for pol, s0 in (("vv", sigma0_vv), ("vh", sigma0_vh)):
            n = f"s1a-iw-grd-{pol}-20220830t000000-001"
            f = tmp_path / f"{n}.tiff"
            with rasterio.open(f, "w", driver="GTiff", height=H, width=W, count=1, dtype="uint16") as o:
                o.gcps = (g, "EPSG:4326")
                o.write((np.sqrt(s0) * A).astype("uint16")[::-1], 1)
            z.write(f, f"X.SAFE/measurement/{n}.tiff")
            z.writestr(f"X.SAFE/annotation/calibration/calibration-{n}.xml",
                       "<calibration><calibrationVectorList>" + "".join(
                           f"<calibrationVector><line>{ln}</line><pixel>0 {W - 1}</pixel><sigmaNought>{A} {A}</sigmaNought>"
                           "</calibrationVector>" for ln in (0, H)) + "</calibrationVectorList></calibration>")
    return zp


def test_sentinel1_grd_zip_is_calibrated_and_georeferenced(tmp_path):
    from resqflood.sar_io import load_any
    rng = np.random.default_rng(0)
    vv = (10 ** (rng.uniform(-14, -6, (120, 160)) / 10)).astype("float32")
    zp = _write_safe(tmp_path, vv, vv / 5)
    d = load_any([str(zp)])
    assert d["crs"] == "EPSG:4326" and d["vh"] is not None
    assert abs(np.nanmedian(10 * np.log10(d["vv"])) - np.median(10 * np.log10(vv))) < 0.5      # true sigma0 kept
    assert abs(np.nanmedian(10 * np.log10(d["vv"] / d["vh"])) - 10 * np.log10(5)) < 0.2       # VV/VH ratio kept


def test_zip_that_is_not_sentinel1_is_refused(tmp_path):
    import zipfile
    from resqflood.sar_io import load_any
    zp = tmp_path / "photos.zip"
    with zipfile.ZipFile(zp, "w") as z:
        z.writestr("a.txt", "hello")
    with pytest.raises(InputError):
        load_any([str(zp)])
