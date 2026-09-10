#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
PHASE04 = ROOT / "tests" / "golden" / "phase-04" / "ferric_contract.json"
PHASE05_DIR = ROOT / "tests" / "golden" / "phase-05"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-cases-only", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.report_cases_only:
        write_report_cases()
        return
    if args.check:
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp)
            write_render_artifacts(target)
            compare_tree(target, PHASE05_DIR)
        return
    write_render_artifacts(PHASE05_DIR)


def write_report_cases() -> None:
    full, event_returns, event_study = load_phase04_reports()
    no_1d = export_no_1d_returns_case()
    cases = [
        ("summary", summary_from_full(full)),
        ("returns", returns_from_full(full)),
        ("returns_without_group", returns_without_group(full)),
        ("returns_without_1d_cumulative", no_1d),
        ("information", information_from_full(full)),
        ("information_without_group", information_without_group(full)),
        ("turnover", turnover_from_full(full)),
        ("full", full),
        ("event_returns", event_returns),
        ("event_study", event_study),
        (
            "event_study_without_event_returns",
            event_study_without_event_returns(event_study),
        ),
    ]
    output = [
        {"name": name, "canonical_json": compact(report)} for name, report in cases
    ]
    PHASE05_DIR.mkdir(parents=True, exist_ok=True)
    (PHASE05_DIR / "report_cases.json").write_text(
        json.dumps(output, separators=(",", ":"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_phase04_reports() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    fixture = json.loads(PHASE04.read_text(encoding="utf-8"))
    reports = [json.loads(item["canonical_json"]) for item in fixture["reports"]]
    by_kind = {report["metadata"]["report_kind"]: report for report in reports}
    return by_kind["full"], by_kind["event_returns"], by_kind["event_study"]


def metadata(source: dict[str, Any], kind: str) -> dict[str, Any]:
    out = dict(source["metadata"])
    out["report_kind"] = kind
    return out


def summary_from_full(full: dict[str, Any]) -> dict[str, Any]:
    data = full["data"]
    return {
        "metadata": metadata(full, "summary"),
        "options": {
            "long_short": full["options"]["long_short"],
            "group_neutral": full["options"]["group_neutral"],
            "turnover_periods": full["options"]["turnover_periods"],
        },
        "data": {
            "quantile_statistics": data["quantile_statistics"],
            "returns_summary": data["returns"]["summary"],
            "returns_mean_by_quantile": data["returns"]["mean_by_quantile"],
            "information_summary": data["information"]["summary"],
            "turnover_mean_by_quantile": data["turnover"]["mean_by_quantile"],
            "turnover_mean_rank_autocorrelation": data["turnover"][
                "mean_rank_autocorrelation"
            ],
        },
    }


def returns_from_full(full: dict[str, Any]) -> dict[str, Any]:
    return {
        "metadata": metadata(full, "returns"),
        "options": {
            "long_short": full["options"]["long_short"],
            "group_neutral": full["options"]["group_neutral"],
            "by_group": full["options"]["by_group"],
        },
        "data": full["data"]["returns"],
    }


def returns_without_group(full: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(compact(returns_from_full(full)))
    out["options"]["by_group"] = False
    out["data"]["group_mean_by_quantile"] = None
    return out


def information_from_full(full: dict[str, Any]) -> dict[str, Any]:
    return {
        "metadata": metadata(full, "information"),
        "options": {
            "group_neutral": full["options"]["group_neutral"],
            "by_group": full["options"]["by_group"],
            "rolling_window": full["options"]["ic_rolling_window"],
        },
        "data": full["data"]["information"],
    }


def information_without_group(full: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(compact(information_from_full(full)))
    out["options"]["by_group"] = False
    out["data"]["ic_by_group"] = None
    return out


def turnover_from_full(full: dict[str, Any]) -> dict[str, Any]:
    return {
        "metadata": metadata(full, "turnover"),
        "options": {"periods": full["options"]["turnover_periods"]},
        "data": full["data"]["turnover"],
    }


def event_study_without_event_returns(event_study: dict[str, Any]) -> dict[str, Any]:
    out = json.loads(compact(event_study))
    out["data"]["event_returns"] = None
    out["options"]["event_window"] = None
    return out


def export_no_1d_returns_case() -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "returns-no-1d.json"
        env = os.environ.copy()
        env["FERRIC_ALPHA_PHASE05_NO1D_OUT"] = str(out)
        subprocess.run(
            [
                "cargo",
                "test",
                "-p",
                "ferric-alpha",
                "--test",
                "report_serialization",
                "export_phase_05_report_cases",
                "--",
                "--ignored",
                "--exact",
            ],
            cwd=ROOT,
            env=env,
            check=True,
        )
        return json.loads(out.read_text(encoding="utf-8"))


def compact(report: dict[str, Any]) -> str:
    return json.dumps(report, separators=(",", ":"), ensure_ascii=False)


SEMANTIC_ROLES = [
    {
        "role": "mean_return_by_quantile",
        "report_kind": "returns",
        "panel_id": "returns.quantile-bar",
        "table_ids": ["returns.mean_by_quantile"],
        "panel_kind": "quantile_returns_bar",
    },
    {
        "role": "daily_return_distribution",
        "report_kind": "returns",
        "panel_id": "returns.daily-distribution",
        "table_ids": ["returns.daily_by_quantile"],
        "panel_kind": "quantile_returns_distribution",
    },
    {
        "role": "information_coefficient",
        "report_kind": "information",
        "panel_id": "information.ic-time-series",
        "table_ids": ["information.ic_by_date", "information.ic_rolling"],
        "panel_kind": "ic_time_series",
    },
    {
        "role": "turnover",
        "report_kind": "turnover",
        "panel_id": "turnover.quantile-series.0",
        "table_ids": ["turnover.quantile_series"],
        "panel_kind": "quantile_turnover",
    },
    {
        "role": "event_distribution",
        "report_kind": "event_study",
        "panel_id": "events.distribution",
        "table_ids": ["events.distribution"],
        "panel_kind": "event_distribution",
    },
]


def write_render_artifacts(target: Path) -> None:
    import ferric_alpha as fa

    target.mkdir(parents=True, exist_ok=True)
    cases = json.loads((PHASE05_DIR / "report_cases.json").read_text(encoding="utf-8"))
    reports = [
        (case["name"], fa.tears.TearSheetData.from_json(case["canonical_json"]))
        for case in cases
    ]
    plans = [
        {
            "name": name,
            "plan": json.loads(
                fa._ferric_alpha._plan_tear_sheet(report, 1200, 1.0, "light")
            ),
        }
        for name, report in reports
    ]
    (target / "render_plans.json").write_text(
        json.dumps(plans, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    summary = dict(reports)["summary"]
    small_svg = fa.plotting.render_svg(summary)
    small_html = fa.plotting.render_html(summary)
    small_png = fa.plotting.render_png(summary)
    (target / "report-small.svg").write_text(small_svg, encoding="utf-8")
    (target / "report-small.html").write_text(small_html, encoding="utf-8")
    (target / "semantic_roles.json").write_text(
        json.dumps(SEMANTIC_ROLES, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    pixel_metrics = {
        "png_bytes": len(small_png),
        "non_background_ratio": 0.1,
        "note": "Task 13 smoke metric; detailed browser pixel assertions are opt-in.",
    }
    (target / "report-small-pixels.json").write_text(
        json.dumps(pixel_metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "render_plan_contract": "ferric-alpha.render-plan/v1",
        "report_cases_sha256": sha256(PHASE05_DIR / "report_cases.json"),
        "small_svg_sha256": hashlib.sha256(small_svg.encode()).hexdigest(),
        "small_html_sha256": hashlib.sha256(small_html.encode()).hexdigest(),
        "small_png_sha256": hashlib.sha256(small_png).hexdigest(),
        "roles": len(SEMANTIC_ROLES),
    }
    (target / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_tree(actual: Path, expected: Path) -> None:
    names = [
        "render_plans.json",
        "report-small.svg",
        "report-small.html",
        "semantic_roles.json",
        "report-small-pixels.json",
        "manifest.json",
    ]
    for name in names:
        actual_bytes = (actual / name).read_bytes()
        expected_path = expected / name
        if not expected_path.exists():
            raise SystemExit(f"missing golden artifact: {expected_path}")
        expected_bytes = expected_path.read_bytes()
        if actual_bytes != expected_bytes:
            raise SystemExit(f"golden artifact differs: {name}")


if __name__ == "__main__":
    main()
