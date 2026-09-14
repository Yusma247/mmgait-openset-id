"""Every path in this project is derived from where this file sits.

Layout expected one level above `code/`:

    <project>/
        mmGait10/            raw dataset (point_clouds/, spectrograms/)
        mmGait10_v2.zip      original archive
        prepared_64pts.npz   built by prep.py
        code/                this folder
        figures/
        results/

Move the whole project folder anywhere and nothing needs editing.
"""
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]

RAW = PROJECT / "mmGait10"
POINT_CLOUDS = RAW / "point_clouds"
SPECTROGRAMS = RAW / "spectrograms"
ZIP = PROJECT / "mmGait10_v2.zip"
PREPARED = PROJECT / "prepared_64pts.npz"

FIGURES = PROJECT / "figures"
RESULTS = PROJECT / "results"
RUNS = RESULTS / "runs"
RUNS_XS = RESULTS / "runs_xs"

for _d in (FIGURES, RESULTS, RUNS, RUNS_XS):
    _d.mkdir(parents=True, exist_ok=True)


def require(p, what):
    if not Path(p).exists():
        raise SystemExit(
            f"Cannot find {what} at:\n  {p}\n"
            f"Expected it under the project root {PROJECT}. "
            f"See the layout in code/paths.py."
        )
    return Path(p)
