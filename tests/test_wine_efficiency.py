from __future__ import annotations

import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from experiments.plot_wine_efficiency_comparison import (
    LEGEND_HANDLE_ORDER,
    build_figure,
    save_figure,
    validate_plot_report,
)
from experiments.run_wine_efficiency_comparison import (
    DEFAULT_INPUT,
    METHOD_LABELS,
    METHOD_ORDER,
    _benchmark_scalar_wine_utility,
    build_efficiency_report,
    validate_efficiency_report,
)


def _source() -> dict:
    if not DEFAULT_INPUT.exists():
        pytest.skip("audited Wine source report is unavailable")
    return json.loads(DEFAULT_INPUT.read_text(encoding="utf-8"))


def _report() -> dict:
    return build_efficiency_report(
        _source(),
        source_path=DEFAULT_INPUT,
        utility_calibration={"median_scalar_seconds_per_call": 1.0e-3},
    )


def test_scalar_wine_utility_calibration_is_deterministic() -> None:
    calibration = _benchmark_scalar_wine_utility(
        _source(), calls_per_block=10, blocks=2, warmup_calls=2, seed=43
    )
    assert calibration["median_scalar_seconds_per_call"] > 0.0
    assert len(calibration["scalar_block_seconds"]) == 2
    assert calibration["coalition_batch_sha256"]


def test_wine_efficiency_charges_greedy_and_sdiff_extra_time() -> None:
    report = _report()
    validate_efficiency_report(report)
    assert len(report["results"]) == 5
    assert report["results"][0]["methods"]["ofa"]["source_method"] == (
        "ofa_iid_ratio"
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
        assert greedy["mean_charged_extra_seconds"] == pytest.approx(
            greedy["mean_charged_design_seconds"]
        )
        assert len(greedy["charged_design_provenance_by_repeat"]) == 3
        sdiff = row["methods"]["s_diff"]
        assert sdiff["mean_charged_design_seconds"] == 0.0
        assert (
            sdiff[
                "mean_charged_sdiff_end_to_end_wall_upper_bound_seconds"
            ]
            > 0.0
        )
        assert sdiff["mean_charged_extra_seconds"] == pytest.approx(
            sdiff[
                "mean_charged_sdiff_end_to_end_wall_upper_bound_seconds"
            ]
        )
        assert len(
            sdiff[
                "charged_sdiff_end_to_end_wall_upper_bound_provenance_by_repeat"
            ]
        ) == 3
        for method in set(METHOD_ORDER) - {"inside_greedy", "s_diff"}:
            assert row["methods"][method]["mean_charged_design_seconds"] == 0.0
            assert row["methods"][method]["mean_charged_extra_seconds"] == 0.0


def test_wine_efficiency_validator_detects_corrupted_time() -> None:
    report = _report()
    corrupted = copy.deepcopy(report)
    corrupted["results"][0]["methods"]["ofa"][
        "mean_estimated_total_seconds"
    ] += 1.0
    with pytest.raises(ValueError, match="mean_estimated_total_seconds"):
        validate_efficiency_report(corrupted)


def test_wine_efficiency_validator_detects_corrupted_sdiff_surcharge() -> None:
    report = _report()
    corrupted = copy.deepcopy(report)
    corrupted["results"][0]["methods"]["s_diff"][
        "charged_sdiff_end_to_end_wall_upper_bound_seconds_by_repeat"
    ][0] += 1.0
    with pytest.raises(ValueError, match="charged extra time"):
        validate_efficiency_report(corrupted)


def test_wine_efficiency_figure_contract_and_artifacts(tmp_path: Path) -> None:
    report = _report()
    validate_plot_report(report)
    figure = build_figure(report)
    axis = figure.axes[0]
    assert axis.get_title() == ""
    assert axis.get_xscale() == axis.get_yscale() == "log"
    assert axis.get_xlabel() == "Estimated time (s)"
    assert axis.get_ylabel() == "RMSE"
    assert all(spine.get_visible() for spine in axis.spines.values())
    assert [text.get_text() for text in axis.get_legend().get_texts()] == [
        METHOD_LABELS[method] for method in LEGEND_HANDLE_ORDER
    ]
    plt.close(figure)

    source = tmp_path / "report.json"
    source.write_text(json.dumps(report), encoding="utf-8")
    png, pdf, svg, metadata = save_figure(
        report, tmp_path / "efficiency.png", source_report=source
    )
    assert png.stat().st_size > 10_000
    assert pdf.read_bytes().startswith(b"%PDF")
    assert "<svg" in svg.read_text(encoding="utf-8")[:500]
    saved = json.loads(metadata.read_text(encoding="utf-8"))
    assert saved["panel_alignment"]["status"] == (
        "not_applicable_single_panel"
    )
    assert np.isclose(saved["time_proxy"]["utility_seconds_per_call"], 1e-3)
    assert "sdiff_surcharge_caveat" in saved["time_proxy"]
