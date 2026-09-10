mod common;

use ferric_alpha::TearSheetData;
use ferric_alpha_render::{
    PanelKind, RenderError, RenderOptions, RenderTheme, plan_report, render_panel_svg, render_svg,
    theme_for,
};
use serde_json::Value;

#[test]
fn event_panels_render_distribution_and_cumulative_paths() {
    for case_name in ["event_returns", "event_study"] {
        let (_, report) = common::phase_05_report_cases()
            .into_iter()
            .find(|(name, _)| name == case_name)
            .unwrap();
        let plan = plan_report(&report, &RenderOptions::default()).unwrap();
        let chart_panels = plan
            .panels
            .iter()
            .filter(|panel| {
                matches!(
                    panel.kind,
                    PanelKind::EventDistribution | PanelKind::EventAverageCumulative { .. }
                )
            })
            .collect::<Vec<_>>();
        assert!(!chart_panels.is_empty());

        for panel in chart_panels {
            let rendered =
                render_panel_svg(&report, panel, 900, theme_for(RenderTheme::Light)).unwrap();
            assert!(rendered.svg.contains(&format!("id=\"{}\"", panel.id)));
            assert!(rendered.svg.contains("data-role=\"plot-body\""));
            assert!(rendered.svg.contains("data-mark=\""));
            assert!(!rendered.svg.contains("NaN"));
            assert!(!rendered.svg.contains("Infinity"));
            roxmltree::Document::parse(&rendered.svg).unwrap();
        }
    }
}

#[test]
fn event_study_full_svg_reuses_event_and_returns_panels() {
    let (_, report) = common::phase_05_report_cases()
        .into_iter()
        .find(|(name, _)| name == "event_study")
        .unwrap();
    let svg = render_svg(&report, &RenderOptions::default()).unwrap();
    assert!(svg.contains("events.distribution"));
    assert!(svg.contains("events.average-cumulative"));
    assert!(svg.contains("events.quantile-bar"));
    assert!(svg.contains("events.daily-distribution"));
    roxmltree::Document::parse(&svg).unwrap();
}

#[test]
fn event_distribution_rejects_overlapping_bins() {
    let (_, report) = common::phase_05_report_cases()
        .into_iter()
        .find(|(name, _)| name == "event_study")
        .unwrap();
    let mut value = serde_json::from_str::<Value>(&report.to_json().unwrap()).unwrap();
    let columns = value["data"]["distribution"]["columns"]
        .as_array_mut()
        .unwrap();
    for column in columns {
        if column["name"] == "bin_start" {
            column["data"]["datetime"]["values"][1] = Value::from(1704205800000_i64);
        }
    }
    let report = TearSheetData::from_json(&serde_json::to_string(&value).unwrap()).unwrap();
    let plan = plan_report(&report, &RenderOptions::default()).unwrap();
    let panel = plan
        .panels
        .iter()
        .find(|panel| matches!(panel.kind, PanelKind::EventDistribution))
        .unwrap();
    let err = render_panel_svg(&report, panel, 720, theme_for(RenderTheme::Light)).unwrap_err();
    assert!(matches!(err, RenderError::InvalidContract { .. }));
}

#[test]
fn event_uncertainty_overflow_is_rejected() {
    let (_, report) = common::phase_05_report_cases()
        .into_iter()
        .find(|(name, _)| name == "event_returns")
        .unwrap();
    let mut value = serde_json::from_str::<Value>(&report.to_json().unwrap()).unwrap();
    let columns = value["data"]["average_cumulative_returns"]["columns"]
        .as_array_mut()
        .unwrap();
    for column in columns {
        if column["name"] == "mean_cumulative_return" {
            column["data"]["float64"][0] = Value::from(f64::MAX);
        }
        if column["name"] == "std_cumulative_return" {
            column["data"]["float64"][0] = Value::from(f64::MAX);
        }
    }
    let report = TearSheetData::from_json(&serde_json::to_string(&value).unwrap()).unwrap();
    let plan = plan_report(&report, &RenderOptions::default()).unwrap();
    let panel = plan
        .panels
        .iter()
        .find(|panel| matches!(panel.kind, PanelKind::EventAverageCumulative { .. }))
        .unwrap();
    let err = render_panel_svg(&report, panel, 900, theme_for(RenderTheme::Light)).unwrap_err();
    assert!(matches!(err, RenderError::InvalidContract { .. }));
}
