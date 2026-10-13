import numpy as np
from rasterio.transform import from_origin
from resqflood.sar import training_data as td


def test_lee_reduces_speckle_and_keeps_mean():
    rng = np.random.default_rng(0)
    clean = np.full((128, 128), 0.1, "float32")
    noisy = clean * rng.gamma(4.4, 1 / 4.4, clean.shape).astype("float32")
    out = td.lee_filter(noisy, np.ones_like(noisy, bool), 7, 4.4)
    assert out.std() < noisy.std() / 3
    assert abs(out.mean() - 0.1) < 0.005


def test_lee_respects_invalid():
    a = np.ones((32, 32), "float32")
    v = np.ones_like(a, bool)
    v[:, :4] = False
    out = td.lee_filter(a, v)
    assert np.isnan(out[:, :4]).all() and np.allclose(out[:, 4:], 1)


def test_labels_permanent_water():
    flood = np.array([[1, 0, 0, 1]], "uint8")
    perm = np.array([[0, 1, 0, 1]], "uint8")
    ok = np.array([[True, True, True, False]])
    lab = td.make_labels(flood, ok, np.ones_like(ok), perm, label_includes_permanent=False)
    water, fl, p, valid = lab[:, 0]
    assert water.tolist() == [1, 1, 0, 255]
    assert fl.tolist() == [1, 255, 0, 255]
    assert p.tolist() == [0, 1, 0, 1] and valid.tolist() == [1, 1, 1, 0]


def test_db_roundtrip():
    db = np.array([-23.456, np.nan], "float32")
    i = td.db_to_i16(db)
    assert i[0] == -2346 and i[1] == td.NODATA_I16
