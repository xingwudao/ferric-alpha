mod common;

use ferric_alpha::TearSheetData;
use ferric_alpha_render::{
    PanelKind, RenderError, RenderOptions, RenderTheme, plan_report, render_panel_svg, render_svg,
    theme_for,
};
use serde_json::Value;

#[test]
fn turnover_panels_render_one_pair_per_period() {
    let (_, report) = common::phase_05_report_cases()
        .into_iter()
        .find(|(name, _)| name == "turnover")
        .unwrap();
    let plan = plan_report(&report, &RenderOptions::default()).unwrap();
    let chart_panels = plan
        .panels
        .iter()
        .filter(|panel| {
            matches!(
                panel.kind,
                PanelKind::QuantileTurnover { .. } | PanelKind::RankAutocorrelation { .. }
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(chart_panels.len(), 4);

    for panel in chart_panels {
        let rendered =
            render_panel_svg(&report, panel, 720, theme_for(RenderTheme::Light)).unwrap();
        assert!(rendered.svg.contains(&format!("id=\"{}\"", panel.id)));
        assert!(rendered.svg.contains("data-role=\"plot-body\""));
        assert!(rendered.svg.contains("data-mark=\"line\""));
        assert!(!rendered.svg.contains("NaN"));
        assert!(!rendered.svg.contains("Infinity"));
        roxmltree::Document::parse(&rendered.svg).unwrap();
    }
}

#[test]
fn turnover_full_svg_includes_turnover_chart_panels() {
    let (_, report) = common::phase_05_report_cases()
        .into_iter()
        .find(|(name, _)| name == "turnover")
        .unwrap();
    let svg = render_svg(&report, &RenderOptions::default()).unwrap();
    assert!(svg.contains("turnover.quantile-series.0"));
    assert!(svg.contains("turnover.rank-autocorrelation.0"));
    assert!(svg.contains("data-mark=\"line\""));
    roxmltree::Document::parse(&svg).unwrap();
}

#[test]
fn turnover_outside_zero_one_is_rejected() {
    let (_, report) = common::phase_05_report_cases()
        .into_iter()
        .find(|(name, _)| name == "turnover")
        .unwrap();
    let mut value = serde_json::from_str::<Value>(&report.to_json().unwrap()).unwrap();
    let columns = value["data"]["quantile_series"]["columns"]
        .as_array_mut()
        .unwrap();
    for column in columns {
        if column["name"] == "turnover" {
            column["data"]["float64"][0] = Value::from(1.01);
        }
    }
    let report = TearSheetData::from_json(&serde_json::to_string(&value).unwrap()).unwrap();
    let plan = plan_report(&report, &RenderOptions::default()).unwrap();
    let panel = plan
        .panels
        .iter()
        .find(|panel| matches!(panel.kind, PanelKind::QuantileTurnover { .. }))
        .unwrap();

    let err = render_panel_svg(&report, panel, 720, theme_for(RenderTheme::Light)).unwrap_err();
    assert!(matches!(err, RenderError::InvalidContract { .. }));
}

#[test]
fn rank_autocorrelation_outside_minus_one_one_is_rejected() {
    let (_, report) = common::phase_05_report_cases()
        .into_iter()
        .find(|(name, _)| name == "turnover")
        .unwrap();
    let mut value = serde_json::from_str::<Value>(&report.to_json().unwrap()).unwrap();
    let columns = value["data"]["rank_autocorrelation_series"]["columns"]
        .as_array_mut()
        .unwrap();
    for column in columns {
        if column["name"] == "autocorrelation" {
            column["data"]["float64"][0] = Value::from(-1.01);
        }
    }
    let report = TearSheetData::from_json(&serde_json::to_string(&value).unwrap()).unwrap();
    let plan = plan_report(&report, &RenderOptions::default()).unwrap();
    let panel = plan
        .panels
        .iter()
        .find(|panel| matches!(panel.kind, PanelKind::RankAutocorrelation { .. }))
        .unwrap();

    let err = render_panel_svg(&report, panel, 720, theme_for(RenderTheme::Light)).unwrap_err();
    assert!(matches!(err, RenderError::InvalidContract { .. }));
}
