mod common;

use ferric_alpha::TearSheetData;
use ferric_alpha_render::{
    PanelKind, RenderError, RenderOptions, RenderTheme, plan_report, render_panel_svg, render_svg,
    theme_for,
};
use serde_json::Value;

#[test]
fn information_panels_render_stable_chart_marks() {
    let (_, report) = common::phase_05_report_cases()
        .into_iter()
        .find(|(name, _)| name == "information")
        .unwrap();
    let plan = plan_report(&report, &RenderOptions::default()).unwrap();
    let chart_panels = plan
        .panels
        .iter()
        .filter(|panel| {
            matches!(
                panel.kind,
                PanelKind::IcTimeSeries
                    | PanelKind::IcHistogram { .. }
                    | PanelKind::IcQq
                    | PanelKind::IcMonthlyHeatmap
                    | PanelKind::IcByGroup
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(chart_panels.len(), 5);

    for panel in chart_panels {
        let first = render_panel_svg(&report, panel, 900, theme_for(RenderTheme::Light)).unwrap();
        let second = render_panel_svg(&report, panel, 900, theme_for(RenderTheme::Light)).unwrap();
        assert_eq!(first.svg, second.svg);
        assert!(first.svg.contains(&format!("id=\"{}\"", panel.id)));
        assert!(first.svg.contains("data-role=\"plot-body\""));
        assert!(first.svg.contains("data-mark=\""));
        assert!(!first.svg.contains("NaN"));
        assert!(!first.svg.contains("Infinity"));
        roxmltree::Document::parse(&first.svg).unwrap();
    }
}

#[test]
fn information_without_group_omits_group_panel() {
    let (_, report) = common::phase_05_report_cases()
        .into_iter()
        .find(|(name, _)| name == "information_without_group")
        .unwrap();
    let plan = plan_report(&report, &RenderOptions::default()).unwrap();
    assert!(
        plan.panels
            .iter()
            .all(|panel| !matches!(panel.kind, PanelKind::IcByGroup))
    );
}

#[test]
fn information_full_svg_includes_chart_panels() {
    let (_, report) = common::phase_05_report_cases()
        .into_iter()
        .find(|(name, _)| name == "information")
        .unwrap();
    let svg = render_svg(&report, &RenderOptions::default()).unwrap();
    assert!(svg.contains("information.ic-time-series"));
    assert!(svg.contains("information.ic-histogram"));
    assert!(svg.contains("information.ic-qq"));
    assert!(svg.contains("information.ic-monthly"));
    roxmltree::Document::parse(&svg).unwrap();
}

#[test]
fn raw_ic_outside_minus_one_to_one_is_rejected() {
    let (_, report) = common::phase_05_report_cases()
        .into_iter()
        .find(|(name, _)| name == "information")
        .unwrap();
    let mut value = serde_json::from_str::<Value>(&report.to_json().unwrap()).unwrap();
    let columns = value["data"]["ic_by_date"]["columns"]
        .as_array_mut()
        .unwrap();
    for column in columns {
        if column["name"] == "ic" {
            column["data"]["float64"][0] = Value::from(1.01);
        }
    }
    let report = TearSheetData::from_json(&serde_json::to_string(&value).unwrap()).unwrap();
    let plan = plan_report(&report, &RenderOptions::default()).unwrap();
    let panel = plan
        .panels
        .iter()
        .find(|panel| matches!(panel.kind, PanelKind::IcTimeSeries))
        .unwrap();

    let err = render_panel_svg(&report, panel, 900, theme_for(RenderTheme::Light)).unwrap_err();
    assert!(matches!(err, RenderError::InvalidContract { .. }));
}
