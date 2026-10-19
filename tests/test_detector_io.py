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
