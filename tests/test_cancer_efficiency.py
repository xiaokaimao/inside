from __future__ import annotations

import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from experiments.plot_cancer_efficiency_comparison import (
    LEGEND_HANDLE_ORDER,
    build_figure,
    save_figure,
    validate_plot_report,
)
from experiments.run_cancer_efficiency_comparison import (
    DEFAULT_INPUT,
    METHOD_LABELS,
    METHOD_ORDER,
    _benchmark_scalar_cancer_utility,
    build_efficiency_report,
    validate_efficiency_report,
)


def _source() -> dict:
    if not DEFAULT_INPUT.exists():
        pytest.skip("audited Cancer source report is unavailable")
    return json.loads(DEFAULT_INPUT.read_text(encoding="utf-8"))


def _report() -> dict:
    return build_efficiency_report(
        _source(),
        source_path=DEFAULT_INPUT,
        utility_calibration={"median_scalar_seconds_per_call": 2.8e-3},
    )


def test_scalar_cancer_utility_calibration_is_deterministic() -> None:
    calibration = _benchmark_scalar_cancer_utility(
        _source(), calls_per_block=8, blocks=2, warmup_calls=2, seed=47
    )
    assert calibration["median_scalar_seconds_per_call"] > 0.0
    assert len(calibration["scalar_block_seconds"]) == 2
    assert calibration["coalition_batch_sha256"]


def test_cancer_efficiency_uses_requested_methods_and_overhead() -> None:
    report = _report()
    validate_efficiency_report(report)
    assert len(report["results"]) == 5
    assert tuple(report["configuration"]["methods"]) == METHOD_ORDER
    assert "s_diff" not in METHOD_ORDER
    assert report["results"][0]["methods"]["ofa"]["source_method"] == (
        "ofa_iid_ratio"
    )
    assert (
        report["results"][0]["methods"]["tmc_shapley"][
            "mean_actual_utility_calls"
        ]
        < report["results"][0]["target_total_utility_calls"]
    )
    omissions = {
        item["method"] for item in report["configuration"]["omitted_source_methods"]
    }
    assert omissions == {"ofa_iid_linear", "s_diff"}
    for row in report["results"]:
        greedy = row["methods"]["inside_greedy"]
        assert greedy["mean_charged_design_seconds"] > 0.0
        assert len(greedy["charged_design_provenance_by_repeat"]) == 3
        for method in METHOD_ORDER[1:]:
            assert row["methods"][method]["mean_charged_design_seconds"] == 0.0


def test_cancer_efficiency_validator_detects_corrupted_time() -> None:
    report = _report()
    corrupted = copy.deepcopy(report)
    corrupted["results"][0]["methods"]["ofa"][
        "mean_estimated_total_seconds"
    ] += 1.0
    with pytest.raises(ValueError, match="mean_estimated_total_seconds"):
        validate_efficiency_report(corrupted)


def test_cancer_efficiency_figure_contract_and_artifacts(tmp_path: Path) -> None:
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
    assert saved["panel_alignment"]["status"] == "not_applicable_single_panel"
    assert np.isclose(saved["time_proxy"]["utility_seconds_per_call"], 2.8e-3)
    omitted = {item["method"]: item for item in saved["omitted_methods"]}
    assert omitted["s_diff"]["actual_oom_observed"] is False
