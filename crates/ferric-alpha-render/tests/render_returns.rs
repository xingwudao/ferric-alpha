mod common;

use ferric_alpha::TearSheetData;
use ferric_alpha_render::{
    PanelKind, RenderError, RenderOptions, RenderTheme, plan_report, render_panel_svg, render_svg,
    theme_for,
};
use serde_json::Value;

#[test]
fn returns_panels_render_deterministic_chart_marks() {
    let (_, report) = common::phase_05_report_cases()
        .into_iter()
        .find(|(name, _)| name == "returns")
        .unwrap();
    let plan = plan_report(
        &report,
        &RenderOptions {
            width: 1200,
            ..RenderOptions::default()
        },
    )
    .unwrap();

    let chart_panels = plan
        .panels
        .iter()
        .filter(|panel| {
            matches!(
                panel.kind,
                PanelKind::QuantileReturnsBar { .. }
                    | PanelKind::QuantileReturnsDistribution { .. }
                    | PanelKind::CumulativeReturns { .. }
                    | PanelKind::MeanSpread
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(chart_panels.len(), 6);

    for panel in chart_panels {
        let first = render_panel_svg(&report, panel, 720, theme_for(RenderTheme::Light)).unwrap();
        let second = render_panel_svg(&report, panel, 720, theme_for(RenderTheme::Light)).unwrap();
        assert_eq!(first.svg, second.svg);
        assert!(first.svg.contains(&format!("id=\"{}\"", panel.id)));
        assert!(first.svg.contains("data-role=\"plot-body\""));
        assert!(first.svg.contains("data-mark=\""));
        assert!(!first.svg.contains("NaN"));
        assert!(!first.svg.contains("inf"));
        roxmltree::Document::parse(&first.svg).unwrap();
    }
}

#[test]
fn returns_without_1d_cumulative_omits_cumulative_panels() {
    let (_, report) = common::phase_05_report_cases()
        .into_iter()
        .find(|(name, _)| name == "returns_without_1d_cumulative")
        .unwrap();
    let plan = plan_report(&report, &RenderOptions::default()).unwrap();
    assert!(
        plan.panels
            .iter()
            .all(|panel| !matches!(panel.kind, PanelKind::CumulativeReturns { .. }))
    );
}

#[test]
fn returns_full_svg_includes_chart_panels() {
    let (_, report) = common::phase_05_report_cases()
        .into_iter()
        .find(|(name, _)| name == "returns")
        .unwrap();
    let svg = render_svg(&report, &RenderOptions::default()).unwrap();
    assert!(svg.contains("returns.quantile-bar"));
    assert!(svg.contains("returns.daily-distribution"));
    assert!(svg.contains("returns.mean-spread"));
    assert!(svg.contains("data-mark=\"bar\""));
    roxmltree::Document::parse(&svg).unwrap();
}

#[test]
fn returns_spread_overflow_is_rejected() {
    let (_, report) = common::phase_05_report_cases()
        .into_iter()
        .find(|(name, _)| name == "returns")
        .unwrap();
    let mut value = serde_json::from_str::<Value>(&report.to_json().unwrap()).unwrap();
    let columns = value["data"]["daily_spread"]["columns"]
        .as_array_mut()
        .unwrap();
    for column in columns {
        if column["name"] == "mean_return_difference" {
            column["data"]["float64"][0] = Value::from(f64::MAX);
        }
        if column["name"] == "joint_std_error" {
            column["data"]["float64"][0] = Value::from(f64::MAX);
        }
    }
    let report = TearSheetData::from_json(&serde_json::to_string(&value).unwrap()).unwrap();
    let plan = plan_report(&report, &RenderOptions::default()).unwrap();
    let panel = plan
        .panels
        .iter()
        .find(|panel| matches!(panel.kind, PanelKind::MeanSpread))
        .unwrap();

    let err = render_panel_svg(&report, panel, 720, theme_for(RenderTheme::Light)).unwrap_err();
    assert!(matches!(err, RenderError::InvalidContract { .. }));
}
