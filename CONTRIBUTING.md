# Working on ResQFlood

1. Pull the latest `main`: `git checkout main && git pull`
2. Make a branch per task: `git checkout -b hasan/export-sar-dates` (format `name/short-task`)
3. Commit small, clear steps: `git commit -m "Add VH band to tiling"`
4. Run `pytest -q` before pushing.
5. Push and open a pull request into `main`; one teammate reviews it.

Rules
- Never commit data files (`.tif`, `.gpkg`, `.zip`, `runs/`). `.gitignore` blocks them.
- Settings go in `config/config.yaml`; your own paths go in `config/local.yaml` (not committed).
- New logic gets a small test in `tests/` using synthetic data.
- Keep every number shown to the panel reproducible by a script in `scripts/`.
