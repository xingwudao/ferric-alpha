from __future__ import annotations

import inspect
import json
import subprocess
import sys
from pathlib import Path

import ferric_alpha as fa

CORE_FUNCTIONS = [
    fa.validate_factor_frame,
    fa.compute_forward_returns,
    fa.quantize_factor,
    fa.get_clean_factor,
    fa.get_clean_factor_and_forward_returns,
    fa.performance.factor_information_coefficient,
    fa.performance.mean_information_coefficient,
    fa.performance.mean_return_by_quantile,
    fa.performance.compute_mean_returns_spread,
    fa.performance.factor_weights,
    fa.performance.factor_returns,
    fa.performance.factor_alpha_beta,
    fa.performance.quantile_turnover,
    fa.performance.factor_rank_autocorrelation,
    fa.tears.create_full_tear_sheet_data,
    fa.plotting.render,
]


def test_core_python_api_has_actionable_docstrings() -> None:
    for function in CORE_FUNCTIONS:
        doc = inspect.getdoc(function)
        assert doc, function.__qualname__
        assert "Parameters\n----------" in doc, function.__qualname__
        assert "Returns\n-------" in doc, function.__qualname__


def test_result_types_explain_their_public_attributes() -> None:
    for result_type, attributes in [
        (fa.CleanFactorResult, ["frame", "loss"]),
        (fa.LossReport, ["input_rows", "output_rows"]),
        (fa.tears.TearSheetData, ["table_ids", "table", "to_json"]),
    ]:
        doc = inspect.getdoc(result_type)
        assert doc, result_type.__name__
        for attribute in attributes:
            assert f"`{attribute}" in doc, (result_type.__name__, attribute)


def test_factor_quickstart_runs_end_to_end(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "examples/factor_quickstart.py",
            "--output-dir",
            str(tmp_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    results = json.loads((tmp_path / "results.json").read_text())
    assert results["case"] == {
        "assets": 20,
        "business_days": 80,
        "groups": 4,
        "periods": ["1D", "5D", "10D"],
    }
    assert results["loss"] == {
        "input_rows": 1600,
        "output_rows": 1400,
        "forward_return_loss": 200,
        "quantile_loss": 0,
    }
    assert all(results["sanity"].values())
    assert results["mean_ic"]["1D"] > 0.4
    assert results["spread"]["1D"] > 0.01

    report = (tmp_path / "factor-report.html").read_text()
    assert report.startswith("<!doctype html>")
    assert "Full Tear Sheet" in report
    assert "Information Coefficient" in report
