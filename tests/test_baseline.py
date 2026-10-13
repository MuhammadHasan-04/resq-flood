import numpy as np

from resqflood.sar.baseline import metrics, otsu_vv


def test_metrics_perfect():
    y = np.array([[True, False], [False, True]]); v = np.ones_like(y)
    assert metrics(y, y, v)["IoU"] == 1.0


def test_otsu_separates_dark_water():
    rng = np.random.default_rng(0)
    label = np.zeros((100, 100), np.uint8); label[:, :40] = 1
    vv = np.where(label == 1, rng.normal(-22, 1, label.shape), rng.normal(-8, 1, label.shape)).astype("float32")
    r = otsu_vv({"vv": vv, "valid": np.ones_like(label, bool), "label": label})
    assert r["IoU"] > 0.95
