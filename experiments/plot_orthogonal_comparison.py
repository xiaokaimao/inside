"""Plot original baselines, ShapDoE and Orthogonal using actual utility calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from experiments.add_orthogonal_baseline import validate_report
from experiments.plot_shapdoe_comparison import ADDED_STYLES, _build_figure, _export_artifacts

ORTHOGONAL_STYLES = ADDED_STYLES | {
    "orthogonal": {"label": "Orthogonal", "color": "#D53E76", "marker": "P",
                   "linestyle": "-", "linewidth": 2.8},
}


def build_figure(report: Mapping[str, Any], *, allow_incomplete: bool = False):
    validate_report(report, require_complete=not allow_incomplete)
    return _build_figure(report, ORTHOGONAL_STYLES,
                        partial_note="Partial comparison; only completed budget points are shown")


def export_report(path: Path, *, allow_incomplete: bool = False) -> tuple[Path, Path, Path]:
    report = json.loads(path.read_text())
    figure = build_figure(report, allow_incomplete=allow_incomplete)
    return _export_artifacts(path, report, figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--allow-incomplete", action="store_true")
    args = parser.parse_args()
    for path in args.reports:
        print(export_report(path, allow_incomplete=args.allow_incomplete))


if __name__ == "__main__":
    main()
