"""Build the DEMO_IMAGES folder: real Sentinel-1 examples of every situation, each with an expected answer.

  python scripts/make_demo_pack.py --data ../outputs/FLOOD_TRAINING_DATA --dadu ../outputs/DADU_DATA --out ../outputs/DEMO_IMAGES

All scenes come from held-out areas/dates (validation / test split or pre-monsoon 2022), never from training chips.
expected.csv lists every case; the app's "Demo set" tab and scripts/acceptance_test.py check the answers.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

# (case, chip id or PRE id, expected, what it shows)
GEOTIFF_CASES = [
    ("A_dadu_flood", "TW0284", "FLOOD DETECTED", "Dadu district, 8 Sep 2022: large flood"),
    ("B_flood_and_river", "TW0138", "FLOOD DETECTED", "Flood next to the Indus: river shown blue, flood red"),
    ("C_river_only_same_place_april", "PRE_TW0138", "NO FLOOD DETECTED", "Same place in April 2022: river only, no flood"),
    ("D_lake_only_april", "PRE_TW0359", "NO FLOOD DETECTED", "Large lake (Keenjhar area) in April: permanent water only"),
    ("E_dry_land", "TW0223", "NO FLOOD DETECTED", "Dry land (validation area), no water"),
    ("F_small_flood", "TW0102", "FLOOD DETECTED", "Small flood (~6 % of the image)"),
    ("G_river_only_april_2", "PRE_TW0216", "NO FLOOD DETECTED", "Another river reach in April 2022"),
]


def find_chip(data, cid):
    if cid.startswith("PRE_"):
        hits = glob.glob(str(Path(data) / "00_raw_downloads" / "sentinel1_rtc_pre_flood_2022" / f"{cid}_*.tif"))
        if hits:
            return hits[0]
        z = Path(data) / "00_raw_downloads" / "sentinel1_rtc_pre_flood_2022" / "pre_flood_chips.zip"
    else:
        hits = glob.glob(str(Path(data) / "00_raw_downloads" / "sentinel1_rtc_chips_tuwien" / f"{cid}_*.tif"))
        if hits:
            return hits[0]
        z = None
    zips = [z] if z else sorted((Path(data) / "00_raw_downloads" / "sentinel1_rtc_chips_tuwien").glob("*.zip"))
    for zp in zips:
        if zp and zp.exists():
            with zipfile.ZipFile(zp) as zf:
                for n in zf.namelist():
                    if n.startswith(cid + "_") and n.endswith(".tif"):
                        tmp = Path(data) / "00_raw_downloads" / "_demo_tmp"
                        tmp.mkdir(exist_ok=True)
                        zf.extract(n, tmp)
                        return str(tmp / n)
    raise FileNotFoundError(cid)


def write_2band(src, dst):
    with rasterio.open(src) as s:
        a = s.read().astype("float32")
        prof = s.profile
    a[a <= -30000] = np.nan
    prof.update(dtype="float32", nodata=np.nan, compress="deflate", predictor=3, count=2)
    with rasterio.open(dst, "w", **prof) as o:
        o.write(a)
        o.set_band_description(1, "VV")
        o.set_band_description(2, "VH")
        o.update_tags(units="linear gamma0 (Sentinel-1 RTC, Microsoft Planetary Computer)")
    return a, prof["transform"]


def write_single(a, prof, dst, band):
    p = dict(prof, count=1)
    with rasterio.open(dst, "w", **p) as o:
        o.write(a, 1)
        o.set_band_description(1, band)


def to_png(vv_lin, dst):
    db = 10 * np.log10(np.clip(np.nan_to_num(vv_lin, nan=1e-6), 1e-6, None))
    g = np.clip((db + 25) / 25 * 255, 0, 255).astype("uint8")
    Image.fromarray(g).save(dst)


def bbox_of(T, shape):
    H, W = shape
    return (round(T.c, 6), round(T.f + T.e * H, 6), round(T.c + T.a * W, 6), round(T.f, 6))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../outputs/FLOOD_TRAINING_DATA")
    ap.add_argument("--dadu", default="../outputs/DADU_DATA")
    ap.add_argument("--out", default="../outputs/DEMO_IMAGES")
    a = ap.parse_args()
    out = Path(a.out)
    g1, g2, g3, g4, g5, g6 = (out / d for d in ("01_sentinel1_geotiff", "02_full_scene_dadu", "03_png_type_coordinates",
                                                 "04_png_with_world_or_bbox_file", "05_pre_flood_comparison",
                                                 "06_invalid_inputs"))
    for d in (g1, g2, g3, g4, g5, g6):
        d.mkdir(parents=True, exist_ok=True)
    rows = []
    arrays = {}
    for case, cid, exp, what in GEOTIFF_CASES:
        src = find_chip(a.data, cid)
        a2, T = write_2band(src, g1 / f"{case}.tif")
        arrays[cid] = (a2, T)
        rows.append(dict(case=case, type="Sentinel-1 GeoTIFF (VV+VH)", files=f"01_sentinel1_geotiff/{case}.tif", location="",
                         pixel_m="20", reference="", expected=exp, what_it_shows=what))
    # separate VV / VH files
    src = find_chip(a.data, "TW0216")
    with rasterio.open(src) as s:
        b = s.read().astype("float32")
        prof = dict(s.profile, dtype="float32", nodata=np.nan, compress="deflate")
    b[b <= -30000] = np.nan
    write_single(b[0], prof, g1 / "H_flood_separate_files_VV.tif", "VV")
    write_single(b[1], prof, g1 / "H_flood_separate_files_VH.tif", "VH")
    rows.append(dict(case="H_flood_separate_VV_VH_files", type="two GeoTIFFs (VV, VH)",
                     files="01_sentinel1_geotiff/H_flood_separate_files_VV.tif;01_sentinel1_geotiff/H_flood_separate_files_VH.tif",
                     location="", pixel_m="20", reference="", expected="FLOOD DETECTED",
                     what_it_shows="VV and VH supplied as two files, paired automatically by name"))
    # full Dadu district scene
    vv = sorted(Path(a.dadu).glob("02_Sentinel1_SAR/**/dadu_S1A_20220830_RTC_VV_dB.tif"))
    vh = sorted(Path(a.dadu).glob("02_Sentinel1_SAR/**/dadu_S1A_20220830_RTC_VH_dB.tif"))
    if vv and vh:
        shutil.copy(vv[0], g2 / "dadu_district_20220830_VV_dB.tif")
        shutil.copy(vh[0], g2 / "dadu_district_20220830_VH_dB.tif")
        rows.append(dict(case="I_full_Dadu_district_30Aug2022", type="large GeoTIFF pair (dB)",
                         files="02_full_scene_dadu/dadu_district_20220830_VV_dB.tif;02_full_scene_dadu/dadu_district_20220830_VH_dB.tif",
                         location="", pixel_m="20", reference="", expected="FLOOD DETECTED",
                         what_it_shows="Whole Dadu district (~9,000 km2) on 30 Aug 2022; Manchar lake and Indus shown as permanent water"))
    # PNGs with typed coordinates
    for case, cid, exp, what in [("J_png_flood_typed_corners", "TW0138", "FLOOD DETECTED", "PNG + corner coordinates typed in"),
                                 ("K_png_river_only_typed_corners", "PRE_TW0138", "NO FLOOD DETECTED", "PNG of the river in April + typed corners"),
                                 ("L_png_flood_centre_point", "TW0284", "FLOOD DETECTED", "PNG + only a centre point (lat, lon) from Google Maps")]:
        arr, T = arrays[cid]
        to_png(arr[0], g3 / f"{case}.png")
        w, s_, e, n = bbox_of(T, arr.shape[1:])
        loc = f"{w} {s_} {e} {n}" if "corners" in case else f"{round((s_ + n) / 2, 5)}, {round((w + e) / 2, 5)}"
        px = 20 if "corners" in case else round((e - w) * 111320 * np.cos(np.deg2rad((s_ + n) / 2)) / arr.shape[2], 2)
        rows.append(dict(case=case, type="PNG + typed location", files=f"03_png_type_coordinates/{case}.png", location=loc,
                         pixel_m=str(px), reference="", expected=exp, what_it_shows=what))
    (g3 / "COORDINATES_TO_TYPE.txt").write_text("\n".join(f"{r['files'].split('/')[-1]}: {r['location']}"
                                                          + (f"   (pixel size {r['pixel_m']} m)" if "centre" in r["case"] else "")
                                                          for r in rows if r["type"] == "PNG + typed location") + "\n")
    # PNG with world file / bbox file
    arr, T = arrays["TW0284"]
    to_png(arr[0], g4 / "M_dadu_flood_worldfile.png")
    (g4 / "M_dadu_flood_worldfile.pgw").write_text(f"{T.a}\n{T.d}\n{T.b}\n{T.e}\n{T.c + T.a / 2}\n{T.f + T.e / 2}\n")
    rows.append(dict(case="M_png_with_world_file", type="PNG + .pgw world file",
                     files="04_png_with_world_or_bbox_file/M_dadu_flood_worldfile.png;04_png_with_world_or_bbox_file/M_dadu_flood_worldfile.pgw",
                     location="", pixel_m="20", reference="", expected="FLOOD DETECTED",
                     what_it_shows="location read automatically from the world file"))
    arr, T = arrays["PRE_TW0359"]
    to_png(arr[0], g4 / "N_lake_only.png")
    (g4 / "N_lake_only.bbox.txt").write_text(" ".join(str(v) for v in bbox_of(T, arr.shape[1:])) + "\n")
    rows.append(dict(case="N_png_lake_with_bbox_file", type="PNG + .bbox.txt",
                     files="04_png_with_world_or_bbox_file/N_lake_only.png;04_png_with_world_or_bbox_file/N_lake_only.bbox.txt",
                     location="", pixel_m="20", reference="", expected="NO FLOOD DETECTED",
                     what_it_shows="lake only; location from <name>.bbox.txt"))
    # pre-flood comparison without any location
    for cid, pre in (("TW0138", "PRE_TW0138"),):
        to_png(arrays[cid][0][0], g5 / "O_flood_no_location.png")
        to_png(arrays[pre][0][0], g5 / "O_flood_no_location_before.png")
    rows.append(dict(case="O_png_flood_with_pre_flood_image", type="PNG + pre-flood PNG (no coordinates)",
                     files="05_pre_flood_comparison/O_flood_no_location.png", location="", pixel_m="20",
                     reference="05_pre_flood_comparison/O_flood_no_location_before.png", expected="FLOOD DETECTED",
                     what_it_shows="no coordinates at all: the April image tells the detector where the river normally is"))
    to_png(arrays["PRE_TW0216"][0][0], g5 / "P_river_no_location.png")
    rows.append(dict(case="P_png_river_no_location_no_reference", type="PNG without any location",
                     files="05_pre_flood_comparison/P_river_no_location.png", location="", pixel_m="20", reference="",
                     expected="ANY", what_it_shows="LIMITATION demo: without location or pre-flood image the river "
                                                   "cannot be told apart from flood (the result warns about it)"))
    # invalid inputs
    rng = np.random.default_rng(0)
    photo = np.zeros((300, 400, 3), "uint8")
    photo[:150] = (110, 160, 230)                                   # sky
    photo[150:] = (60, 140, 60)                                     # field
    photo[220:260, 50:350] = (90, 110, 160)                         # water
    photo = np.clip(photo + rng.normal(0, 12, photo.shape), 0, 255).astype("uint8")
    Image.fromarray(photo).save(g6 / "Q_optical_colour_photo.jpg")
    rows.append(dict(case="Q_optical_colour_photo", type="colour photo (not radar)", files="06_invalid_inputs/Q_optical_colour_photo.jpg",
                     location="", pixel_m="20", reference="", expected="REJECTED",
                     what_it_shows="a normal colour picture is refused with an explanation (not radar)"))
    ms = (rng.random((4, 200, 200)) * 3000).astype("uint16")
    with rasterio.open(g6 / "R_optical_multispectral.tif", "w", driver="GTiff", height=200, width=200, count=4, dtype="uint16",
                       crs="EPSG:4326", transform=rasterio.transform.from_origin(67.5, 26.8, 0.0002, 0.0002)) as o:
        o.write(ms)
    rows.append(dict(case="R_optical_multispectral_raster", type="4-band optical GeoTIFF", files="06_invalid_inputs/R_optical_multispectral.tif",
                     location="", pixel_m="20", reference="", expected="REJECTED",
                     what_it_shows="optical satellite rasters are rejected with an explanation"))
    Image.fromarray(np.full((100, 100), 128, "uint8")).save(g6 / "S_empty_image.png")
    rows.append(dict(case="S_empty_image", type="blank PNG", files="06_invalid_inputs/S_empty_image.png", location="",
                     pixel_m="20", reference="", expected="REJECTED", what_it_shows="empty / constant images are rejected"))
    with open(out / "expected.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out / "README.md").write_text(README)
    tmp = Path(a.data) / "00_raw_downloads" / "_demo_tmp"
    if tmp.exists():
        shutil.rmtree(tmp, ignore_errors=True)
    print(f"{len(rows)} demo cases written to {out}")
    print(json.dumps([[r["case"], r["expected"]] for r in rows], indent=0))


README = """# ResQFlood demo images

Real Sentinel-1 radar scenes (Copernicus Sentinel data 2022, RTC from Microsoft Planetary Computer) from areas and dates
that were NOT used to train the model, plus a few invalid inputs. `expected.csv` lists the expected answer of each case;
the app's "Demo set" tab and `scripts/acceptance_test.py` check them automatically.

| Folder | What to upload in the app |
|---|---|
| 01_sentinel1_geotiff | any .tif (2 bands VV+VH), or both H_..._VV.tif and H_..._VH.tif together |
| 02_full_scene_dadu | both files together (whole Dadu district, 30 Aug 2022) |
| 03_png_type_coordinates | a .png, then type its location from COORDINATES_TO_TYPE.txt in the table |
| 04_png_with_world_or_bbox_file | the .png together with its .pgw or .bbox.txt |
| 05_pre_flood_comparison | O_flood_no_location.png as image + O_flood_no_location_before.png as pre-flood image |
| 06_invalid_inputs | to show that wrong inputs are handled |
"""

if __name__ == "__main__":
    main()
