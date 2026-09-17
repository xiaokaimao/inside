from __future__ import annotations

import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from experiments.plot_airport_efficiency_comparison import (
    build_figure,
    save_figure,
    validate_plot_report,
)
from experiments.run_airport_efficiency_comparison import (
    DEFAULT_ABLATION,
    DEFAULT_BASELINES,
    METHOD_LABELS,
    METHOD_ORDER,
    _benchmark_scalar_airport_utility,
    build_efficiency_report,
    validate_efficiency_report,
)


def _real_report(tmp_path: Path) -> dict:
    if not DEFAULT_ABLATION.exists() or not DEFAULT_BASELINES.exists():
        pytest.skip("audited Airport source reports are not available")
    ablation = json.loads(DEFAULT_ABLATION.read_text(encoding="utf-8"))
    baselines = json.loads(DEFAULT_BASELINES.read_text(encoding="utf-8"))
    return build_efficiency_report(
        ablation,
        baselines,
        ablation_path=DEFAULT_ABLATION,
        baseline_path=DEFAULT_BASELINES,
        utility_calibration={"median_scalar_seconds_per_call": 1.0e-5},
    )


def test_scalar_airport_utility_calibration_is_consistent() -> None:
    calibration = _benchmark_scalar_airport_utility(
        calls_per_block=40,
        blocks=2,
        warmup_calls=5,
        seed=41,
    )
    assert calibration["median_scalar_seconds_per_call"] > 0.0
    assert len(calibration["scalar_block_seconds"]) == 2
    assert calibration["coalition_batch_sha256"]


def test_efficiency_report_uses_actual_calls_and_greedy_overhead(
    tmp_path: Path,
) -> None:
    report = _real_report(tmp_path)
    validate_efficiency_report(report)
    assert len(report["results"]) == 5
    assert report["results"][0]["methods"]["ofa"]["source_method"] == (
        "official_ofa_fixed_ratio"
    )
    assert (
        report["results"][0]["methods"]["tmc_shapley"][
            "mean_actual_utility_calls"
        ]
        < report["results"][0]["target_total_utility_calls"]
    )
    for row in report["results"]:
        greedy = row["methods"]["inside_greedy"]
        assert greedy["mean_charged_design_seconds"] > 0.0
        assert len(greedy["charged_design_provenance_by_repeat"]) == 3
        for method in METHOD_ORDER[1:]:
            assert row["methods"][method]["mean_charged_design_seconds"] == 0.0


def test_efficiency_validator_detects_corrupted_means(tmp_path: Path) -> None:
    report = _real_report(tmp_path)
    corrupted = copy.deepcopy(report)
    corrupted["results"][0]["methods"]["ofa"][
        "mean_actual_utility_calls"
    ] += 1.0
    with pytest.raises(ValueError, match="mean_actual_utility_calls"):
        validate_efficiency_report(corrupted)


def test_efficiency_figure_contract_and_artifacts(tmp_path: Path) -> None:
    report = _real_report(tmp_path)
    validate_plot_report(report)
    figure = build_figure(report)
    axis = figure.axes[0]
    assert axis.get_title() == ""
    assert axis.get_xscale() == axis.get_yscale() == "log"
    assert axis.get_xlabel() == "Estimated time (s)"
    assert axis.get_ylabel() == "RMSE"
    assert all(spine.get_visible() for spine in axis.spines.values())
    assert [text.get_text() for text in axis.get_legend().get_texts()] == [
        METHOD_LABELS[method] for method in METHOD_ORDER
    ]
    plt.close(figure)

    source = tmp_path / "report.json"
    source.write_text(json.dumps(report), encoding="utf-8")
    png, pdf, svg, metadata = save_figure(
        report,
        tmp_path / "efficiency.png",
        source_report=source,
    )
    assert png.stat().st_size > 10_000
    assert pdf.read_bytes().startswith(b"%PDF")
    assert "<svg" in svg.read_text(encoding="utf-8")[:500]
    saved = json.loads(metadata.read_text(encoding="utf-8"))
    assert saved["panel_alignment"] == "not_applicable_single_panel"
    assert np.isclose(
        saved["time_proxy"]["utility_seconds_per_call"], 1.0e-5
    )
