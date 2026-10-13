"""Check that DADU_DATA is found and every file the pipeline needs exists.  python scripts/check_data.py"""
import _common  # noqa: F401
from resqflood.config import load_config

cfg = load_config()
need = [cfg["study_area"]["boundary"], cfg["study_area"]["pilot_sector"], cfg["roads"]["file"],
        cfg["flood"]["permanent_water"], cfg["boat"]["dem"], cfg["boat"]["slope"], cfg["boat"]["jrc_occurrence"]]
need += [f"{cfg['flood']['daily_masks_dir']}/dadu_tuwien_flood_{d}.tif" for d in cfg["flood"]["dates"]]
print("DADU_DATA:", cfg.data_root, "found" if cfg.data_root.exists() else "NOT FOUND - set paths.dadu_data in config/local.yaml")
missing = [p for p in need if not cfg.data(p).exists()]
print(f"{len(need) - len(missing)}/{len(need)} required files present")
for p in missing:
    print("  missing:", p)
raise SystemExit(1 if missing else 0)
