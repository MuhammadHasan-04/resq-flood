"""Load any SAR input for the detector and return LINEAR backscatter on a ~20 m grid (the training resolution).

Supported
  * GeoTIFF / raster: 2 bands (VV, VH) in one file, two single-band files (…VV… / …VH…), or a single VV band.
    Units detected automatically: linear power, dB, dB x 100 (int16), uncalibrated 16-bit DN amplitude.
    Any coordinate system (reprojected to WGS84 lon/lat).
  * PNG / JPG (8- or 16-bit, grey or colour quick-look): location from
      - corner coordinates  "west south east north"   (e.g. 67.60 26.60 67.80 26.80)
      - or a centre point   "lat, lon"                 (as copied from Google Maps) + the pixel size
      - or a sidecar file next to the image: world file (.pgw/.jgw/.wld) or <name>.bbox.txt with the 4 corners
    Without a location the image still works, but permanent water cannot be removed (unless a pre-flood image is given).
Rejected with a clear message: optical multispectral rasters (3+ bands without VV/VH), empty / all-nodata images.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

TARGET_DEG = 0.00018          # ~20 m
MAX_PIXELS = 36_000_000       # larger scenes are resampled more coarsely to fit in memory
IMG_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
WORLD_EXT = (".pgw", ".jgw", ".pngw", ".jpgw", ".wld", ".bpw")


class InputError(ValueError):
    pass


def _to_linear(a, dtype, dn_scale=None):
    """dn_scale: for uncalibrated DN pairs the VH band must use the SAME scale as VV (keeps the VV/VH ratio)."""
    a = a.astype("float32")
    bad = ~np.isfinite(a) | (a <= -30000)
    v = a[~bad]
    if v.size == 0 or np.nanmax(v) == np.nanmin(v):
        raise InputError("the image is empty (no valid pixels or a single constant value)")
    med = float(np.median(v))
    if dtype in ("int16", "int32") and med < -100:
        lin, unit = 10 ** (a / 1000.0), "dB x 100"
    elif med < 0:
        lin, unit = 10 ** (a / 10.0), "dB"
    elif dtype == "uint8":
        lin, unit = 10 ** ((-25 + a / 255.0 * 25) / 10.0), "8-bit display image (assumed -25..0 dB)"
    elif dtype in ("uint16", "int32", "uint32") and np.percentile(v, 99) > 50:
        inten = a ** 2                                         # DN amplitude -> intensity
        k = dn_scale if dn_scale else 10 ** (-9.0 / 10) / max(float(np.median(inten[~bad & (a > 0)])), 1e-12)
        lin, unit = inten * k, "uncalibrated 16-bit amplitude (auto-scaled so typical land = -9 dB VV; approximate)"
        _to_linear.last_dn_scale = k
    else:
        lin, unit = a, "linear power"
    lin = lin.astype("float32")
    lin[bad | (lin <= 0)] = np.nan
    return lin, unit


# ------------------------------------------------------------------ raw Sentinel-1 GRD (SAFE zip / measurement tiff)
def _s1_cal_xml_for(tif_path):
    """Calibration XML text for a measurement tiff inside a SAFE folder (on disk or inside a /vsizip/ path)."""
    import zipfile
    p = str(tif_path)
    name = Path(p).name
    cal_name = "calibration-" + Path(name).with_suffix(".xml").name
    if p.startswith("/vsizip/"):
        zpath, inner = p[len("/vsizip/"):].split(".zip/", 1)
        with zipfile.ZipFile(zpath + ".zip") as z:
            hit = [n for n in z.namelist() if n.endswith("/annotation/calibration/" + cal_name)]
            return z.read(hit[0]).decode() if hit else None
    f = Path(p).parent.parent / "annotation" / "calibration" / cal_name
    return f.read_text() if f.exists() else None


def _s1_sigma0_lut(xml_text, H_full, W_full, Hs, Ws):
    """sigma0 calibration table A on the (possibly reduced) grid: sigma0 = DN^2 / A^2."""
    import xml.etree.ElementTree as ET
    from scipy.interpolate import RegularGridInterpolator
    root = ET.fromstring(xml_text)
    lines, vals, pix = [], [], None
    for v in root.iter("calibrationVector"):
        lines.append(int(v.find("line").text))
        px = np.array(v.find("pixel").text.split(), float)
        sv = np.array(v.find("sigmaNought").text.split(), float)
        if pix is None:
            pix = px
        if len(sv) != len(pix):
            sv = np.interp(pix, px, sv)
        vals.append(sv)
    lines = np.array(lines, float)
    order = np.argsort(lines)
    f = RegularGridInterpolator((lines[order], pix), np.array(vals)[order], bounds_error=False, fill_value=None)
    rr = (np.arange(Hs) + 0.5) * H_full / Hs
    cc = (np.arange(Ws) + 0.5) * W_full / Ws
    R, C = np.meshgrid(rr, cc, indexing="ij")
    return f(np.stack([R.ravel(), C.ravel()], -1)).reshape(Hs, Ws).astype("float32")


def expand_safe_zip(path):
    """A Sentinel-1 GRD .zip (SAFE) -> /vsizip/ paths of its VV and VH measurement tiffs (VV first)."""
    import zipfile
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if "/measurement/" in n and n.lower().endswith((".tiff", ".tif"))]
    if not names:
        raise InputError(f"{Path(path).name} is a zip file but not a Sentinel-1 SAFE product (no measurement/*.tiff). "
                         "Unzip it and upload the GeoTIFF(s), or upload the original Sentinel-1 GRD .zip.")
    def pol(n):
        n = Path(n).name.lower()
        return next((k for k in ("vv", "hh", "vh", "hv") if f"-{k}-" in n), "zz")
    names = sorted(names, key=lambda n: ("vv", "hh", "vh", "hv", "zz").index(pol(n)))
    if "grd" not in Path(names[0]).name.lower():
        raise InputError("only Sentinel-1 GRD products are supported (this SAFE looks like SLC or another level)")
    return [f"/vsizip/{Path(path).resolve()}/{n}" for n in names[:2]]


# ------------------------------------------------------------------ rasters
def load_geotiff(paths):
    import rasterio
    from rasterio.warp import Resampling, calculate_default_transform, reproject
    paths = [str(p) for p in paths]
    notes = []
    if len(paths) >= 2:
        srcs = paths if paths[0].startswith("/vsizip/") else \
            sorted(paths, key=lambda p: ("vh" in Path(p).name.lower() or "hv" in Path(p).name.lower(), p))   # VV first
        bands = [(srcs[0], 1), (srcs[1], 1)]
        notes.append(f"VV = {Path(srcs[0]).name}, VH = {Path(srcs[1]).name}")
    else:
        with rasterio.open(paths[0]) as s:
            n = s.count
            desc = [(d or "").lower() for d in s.descriptions]
            dt = s.dtypes[0]
        has_pol = any(("vv" in d or "vh" in d) for d in desc)
        if n >= 3 and not has_pol:
            raise InputError(f"{Path(paths[0]).name} has {n} bands ({dt}) and no VV/VH band names: it looks like an "
                             "OPTICAL image (e.g. RGB / Sentinel-2 / Landsat). This detector reads RADAR (Sentinel-1) "
                             "backscatter; please give a Sentinel-1 VV/VH image.")
        if n >= 2:
            vv_i = next((i for i, d in enumerate(desc) if "vv" in d), 0) + 1
            vh_i = next((i for i, d in enumerate(desc) if "vh" in d), 1) + 1
            bands = [(paths[0], vv_i), (paths[0], vh_i)]
        else:
            bands = [(paths[0], 1)]
    out, T, crs = [], None, None
    dn_scale = None
    for p, b in bands:
        with rasterio.open(p) as s:
            k = int(np.ceil(np.sqrt(s.width * s.height / (2 * MAX_PIXELS)))) if s.width * s.height > 2 * MAX_PIXELS else 1
            if k > 1:       # very large file (e.g. a full Sentinel-1 GRD, ~25000 x 17000 px): read it pre-averaged
                from rasterio.enums import Resampling as R_
                arr = s.read(b, out_shape=(s.height // k, s.width // k), resampling=R_.average).astype("float32")
                notes.append(f"{Path(p).name}: very large file, read at 1/{k} resolution")
            else:
                arr = s.read(b).astype("float32")
            if s.nodata is not None:
                arr[arr == s.nodata] = np.nan
            Hs, Ws = arr.shape
            cal = _s1_cal_xml_for(p) if (s.crs is None and s.gcps[0] and s.dtypes[b - 1] == "uint16") else None
            if cal:          # raw Sentinel-1 GRD with its calibration table -> true sigma0
                A = _s1_sigma0_lut(cal, s.height, s.width, Hs, Ws)
                lin = np.where(arr > 0, arr.astype("float32") ** 2 / A ** 2, np.nan).astype("float32")
                unit = "Sentinel-1 GRD digital numbers, calibrated to sigma0 with the product's calibration table"
            else:
                _to_linear.last_dn_scale = None
                lin, unit = _to_linear(arr, s.dtypes[b - 1], dn_scale)
                dn_scale = dn_scale or _to_linear.last_dn_scale        # VV (first band) sets the scale for VH
                if dn_scale and s.crs is None and s.gcps[0]:
                    notes.append("WARNING: raw Sentinel-1 GRD file without its calibration file: brightness is only "
                                 "approximately calibrated and results may be unreliable. Upload the complete GRD .zip "
                                 "(SAFE) or a calibrated sigma0 GeoTIFF instead.")
            gcps, gcrs = s.gcps if s.crs is None else ([], None)
            if k > 1 and gcps:
                from rasterio.control import GroundControlPoint
                gcps = [GroundControlPoint(row=g.row * Hs / s.height, col=g.col * Ws / s.width, x=g.x, y=g.y, z=g.z)
                        for g in gcps]
            if s.crs is None and not gcps:
                notes.append(f"{Path(p).name}: raster without a coordinate system; treated as a plain image")
                if lin.size > MAX_PIXELS:                   # keep memory bounded on a laptop
                    k = int(np.ceil(np.sqrt(lin.size / MAX_PIXELS)))
                    Hk, Wk = lin.shape[0] // k * k, lin.shape[1] // k * k
                    lin = np.nanmean(lin[:Hk, :Wk].reshape(Hk // k, k, Wk // k, k), axis=(1, 3))
                    notes.append(f"Large image: averaged {k}x{k} pixels to fit in memory")
                out.append(lin)
                continue
            if gcps:        # e.g. a raw Sentinel-1 GRD measurement file (radar geometry + ground control points)
                geo = dict(gcps=gcps, src_crs=gcrs)
                notes.append(f"{Path(p).name}: georeferenced from its {len(gcps)} ground control points (raw GRD)")
                dims = dict(gcps=gcps)
            else:
                geo = dict(src_transform=s.transform * s.transform.scale(s.width / Ws, s.height / Hs), src_crs=s.crs)
                dims = {}
            src_crs = gcrs or s.crs
            bnds = () if gcps else tuple(s.bounds)
            res = TARGET_DEG
            T0, W0, H0 = calculate_default_transform(src_crs, "EPSG:4326", Ws, Hs, *bnds, resolution=res, **dims)
            if W0 * H0 > MAX_PIXELS:
                res = TARGET_DEG * float(np.sqrt(W0 * H0 / MAX_PIXELS))
                notes.append(f"Large scene: processed at {res * 111000:.0f} m pixels to fit in memory")
            T, W, H = calculate_default_transform(src_crs, "EPSG:4326", Ws, Hs, *bnds, resolution=res, **dims)
            dst = np.full((H, W), np.nan, "float32")
            reproject(lin, dst, dst_transform=T, dst_crs="EPSG:4326",
                      resampling=Resampling.average, src_nodata=np.nan, dst_nodata=np.nan, **geo)
            crs = "EPSG:4326"
            out.append(dst)
            notes.append(f"{Path(p).name} band {b}: units {unit}")
    vv = out[0]
    vh = out[1] if len(out) > 1 else None
    if vh is not None and vh.shape != vv.shape:
        notes.append("VV and VH grids differ; VH ignored")
        vh = None
    if crs is None:
        T = None
    return dict(vv=vv, vh=vh, transform=T, crs=crs, kind="raster", notes=notes)


# ------------------------------------------------------------------ PNG / JPG
def parse_location(text):
    """'W S E N' (4 numbers) -> ('bbox', [...]);  'lat, lon' (2 numbers) -> ('center', (lat, lon));  '' -> None."""
    if not text or not str(text).strip():
        return None
    nums = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", str(text))]
    if len(nums) == 4:
        w, s, e, n = nums
        if not (-180 <= w < e <= 180 and -90 <= s < n <= 90):
            raise InputError("corners must be: west south east north (degrees), with west < east and south < north")
        return ("bbox", nums)
    if len(nums) == 2:
        lat, lon = nums
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise InputError("centre point must be: latitude, longitude (degrees)")
        return ("center", (lat, lon))
    raise InputError("location: give 4 numbers (west south east north) or 2 numbers (latitude, longitude)")


def _sidecar_location(path):
    from rasterio.transform import Affine
    p = Path(path)
    for ext in WORLD_EXT + (p.suffix[:2] + p.suffix[-1] + "w", p.suffix + "w"):
        f = p.with_suffix(ext)
        if f.exists():
            a, d, b, e, c, f_ = [float(x) for x in f.read_text().split()[:6]]
            return ("world", Affine(a, b, c - a / 2 - b / 2, d, e, f_ - d / 2 - e / 2))
    for f in (p.with_suffix(".bbox.txt"), p.with_name(p.stem + ".bbox.txt"), p.with_suffix(".txt")):
        if f.exists():
            try:
                return parse_location(f.read_text())
            except InputError:
                pass
    return None


def _read_grey(path):
    from PIL import Image
    im = Image.open(path)
    notes = []
    if im.mode in ("I;16", "I;16B", "I", "F"):
        g = np.asarray(im).astype("float32")
        top = 65535.0 if g.max() > 255 else 255.0      # fixed scale: a contrast stretch would change the radar levels
        g = np.clip(g / top * 255, 0, 255)
        notes.append("16-bit image scaled to 8-bit with a fixed scale (full range = -25..0 dB)")
        return g, notes
    rgb = np.asarray(im.convert("RGB")).astype("float32")
    if im.mode in ("RGBA", "LA", "P"):
        alpha = np.asarray(im.convert("RGBA"))[..., 3]
    else:
        alpha = None
    sat = float(np.mean(np.abs(rgb[..., 0] - rgb[..., 1]) + np.abs(rgb[..., 1] - rgb[..., 2])))
    if sat > 20:
        raise InputError("this is a COLOUR image (a normal photo, Google-Earth/optical satellite picture or a "
                         "false-colour composite). The detector reads radar (Sentinel-1) backscatter, so it cannot "
                         "judge floods in colour photos. Use a grey Sentinel-1 VV image (PNG) or a VV/VH GeoTIFF.")
    if sat > 6:
        notes.append("WARNING: the image has a slight colour tint; it was converted to grey.")
    g = rgb.mean(-1)
    if alpha is not None:
        g[alpha == 0] = np.nan
    return g, notes


def load_image(path, pixel_m=20.0, location=None):
    from PIL import Image
    from rasterio.transform import from_bounds, from_origin
    from rasterio.warp import Resampling, reproject
    g, notes = _read_grey(path)
    loc = parse_location(location) if isinstance(location, str) else location
    if loc is None:
        loc = _sidecar_location(path)
        if loc:
            notes.append(f"Location read from a sidecar file ({loc[0]})")
    H0, W0 = g.shape
    if loc is not None:
        if loc[0] == "world":
            T = loc[1]
            w, n = T.c, T.f
            e, s = T.c + T.a * W0, T.f + T.e * H0
        elif loc[0] == "center":
            lat, lon = loc[1]
            half_w = W0 * pixel_m / 2 / (111320 * np.cos(np.deg2rad(lat)))
            half_h = H0 * pixel_m / 2 / 110540
            w, e, s, n = lon - half_w, lon + half_w, lat - half_h, lat + half_h
            notes.append(f"Image centred on {lat:.5f}, {lon:.5f} with {pixel_m:.0f} m pixels")
        else:
            w, s, e, n = loc[1]
        g8 = np.nan_to_num(g, nan=0)
        lin, unit = _to_linear(g8.astype("uint8"), "uint8")
        lin[~np.isfinite(g)] = np.nan
        T0 = from_bounds(w, s, e, n, W0, H0)
        W, H = max(32, int(round((e - w) / TARGET_DEG))), max(32, int(round((n - s) / TARGET_DEG)))
        if W * H > MAX_PIXELS:
            f = float(np.sqrt(W * H / MAX_PIXELS))
            W, H = int(W / f), int(H / f)
            notes.append("Large area: processed more coarsely to fit in memory")
        T = from_origin(w, n, (e - w) / W, (n - s) / H)
        dst = np.full((H, W), np.nan, "float32")
        reproject(lin, dst, src_transform=T0, src_crs="EPSG:4326", dst_transform=T, dst_crs="EPSG:4326",
                  resampling=Resampling.bilinear, src_nodata=np.nan, dst_nodata=np.nan)
        km = (e - w) * 111.32 * np.cos(np.deg2rad((s + n) / 2))
        notes.append(f"Image placed at W {w:.4f}, S {s:.4f}, E {e:.4f}, N {n:.4f} (~{km:.1f} km wide)")
        notes.append("PNG/JPG: single band, brightness mapped to -25..0 dB VV (approximate).")
        return dict(vv=dst, vh=None, transform=T, crs="EPSG:4326", kind="image+location", notes=notes)
    # no location: resample to ~20 m using the given pixel size
    if pixel_m and abs(pixel_m - 20) > 1:
        f = pixel_m / 20.0
        g = np.asarray(Image.fromarray(np.nan_to_num(g).astype("float32"), mode="F")
                       .resize((max(32, int(W0 * f)), max(32, int(H0 * f))), Image.BILINEAR))
        notes.append(f"Resampled from {pixel_m:.0f} m to 20 m pixels")
    if g.size > 16_000_000:
        f = np.sqrt(16_000_000 / g.size)
        g = np.asarray(Image.fromarray(g.astype("float32"), mode="F").resize((int(g.shape[1] * f), int(g.shape[0] * f))))
        notes.append("Large image downsampled")
    lin, unit = _to_linear(np.clip(np.nan_to_num(g), 0, 255).astype("uint8"), "uint8")
    notes.append("PNG/JPG without location: brightness mapped to -25..0 dB VV (approximate).")
    return dict(vv=lin, vh=None, transform=None, crs=None, kind="image", notes=notes)


def load_any(paths, pixel_m=20.0, location=None):
    """paths: raster(s), or one PNG/JPG (+ optional world file). location: text or (kind, value) for PNG/JPG."""
    paths = [str(p) for p in paths if p]
    if not paths:
        raise InputError("no input file")
    world = [p for p in paths if Path(p).suffix.lower() in WORLD_EXT]
    data_files = [p for p in paths if p not in world and not p.lower().endswith(".txt")]
    if not data_files:
        raise InputError("no image file among the uploads")
    ext = Path(data_files[0]).suffix.lower()
    for f in data_files:
        if Path(f).exists() and Path(f).stat().st_size == 0:
            raise InputError(f"{Path(f).name} is an empty file (0 bytes)")
    try:
        return _load_any(data_files, world, ext, pixel_m, location)
    except InputError:
        raise
    except Exception as e:      # unreadable / corrupted / wrong format -> clear message instead of a crash
        raise InputError(f"{Path(data_files[0]).name} could not be read as an image ({type(e).__name__}: "
                         f"{str(e)[:120]}). Is the file complete and really a GeoTIFF/PNG/JPG?")


def _load_any(data_files, world, ext, pixel_m, location):
    if ext == ".zip":
        return load_geotiff(expand_safe_zip(data_files[0]))
    if ext in (".tif", ".tiff", ".img", ".jp2", ".vrt"):
        return load_geotiff(data_files[:2])
    if ext in IMG_EXT:
        if world and location in (None, ""):
            import shutil
            dst = Path(data_files[0]).with_suffix(".pgw")
            if Path(world[0]) != dst:
                shutil.copy(world[0], dst)
        return load_image(data_files[0], pixel_m, location if location not in ("", None) else None)
    raise InputError(f"unsupported file type '{ext}'. Use GeoTIFF (.tif), a Sentinel-1 GRD .zip or PNG/JPG.")


def align_to(ref, target):
    """Put a reference (pre-flood) input on the grid of the target input. Returns (vv, vh, note)."""
    if ref["transform"] is not None and target["transform"] is not None:
        from rasterio.warp import Resampling, reproject
        out = []
        for k in ("vv", "vh"):
            if ref[k] is None:
                out.append(None)
                continue
            dst = np.full(target["vv"].shape, np.nan, "float32")
            reproject(ref[k], dst, src_transform=ref["transform"], src_crs=ref["crs"],
                      dst_transform=target["transform"], dst_crs=target["crs"], resampling=Resampling.bilinear,
                      src_nodata=np.nan, dst_nodata=np.nan)
            out.append(dst)
        return out[0], out[1], "Pre-flood image aligned by its coordinates"
    from PIL import Image
    H, W = target["vv"].shape
    out = []
    for k in ("vv", "vh"):
        a = ref[k]
        if a is None:
            out.append(None)
            continue
        r = np.asarray(Image.fromarray(np.nan_to_num(a, nan=0).astype("float32"), mode="F").resize((W, H))).astype("float32")
        r[r <= 0] = np.nan
        out.append(r)
    return out[0], out[1], "Pre-flood image has no location: assumed to cover the same area and stretched to the same size"
