mod common;

use std::collections::HashSet;

use ferric_alpha::ReportKind;
use ferric_alpha_render::{
    PanelKind, PanelPlan, RENDER_PLAN_VERSION, RenderError, RenderOptions, RenderPlan, RenderTheme,
    plan_report,
};

#[test]
fn render_plan_serializes_with_stable_contract_and_tagged_panel_kinds() {
    let plan = RenderPlan {
        contract_version: RENDER_PLAN_VERSION.to_string(),
        report_kind: ReportKind::Returns,
        title: "Returns Tear Sheet".to_string(),
        width: 1200,
        height: 424,
        theme: RenderTheme::Light,
        panels: vec![PanelPlan {
            id: "returns.quantile-bar".to_string(),
            title: "Mean Return by Quantile".to_string(),
            row: 0,
            column: 0,
            column_span: 6,
            height: 320,
            table_ids: vec!["returns.mean_by_quantile".to_string()],
            kind: PanelKind::QuantileReturnsBar { by_group: false },
        }],
    };

    let json = plan.to_json().unwrap();
    assert_eq!(plan.to_json().unwrap(), json);
    assert!(json.starts_with(
        "{\"contract_version\":\"ferric-alpha.render-plan/v1\",\"report_kind\":\"returns\""
    ));
    assert!(json.contains("\"kind\":{\"type\":\"quantile_returns_bar\",\"by_group\":false}"));
    assert_eq!(serde_json::from_str::<RenderPlan>(&json).unwrap(), plan);
}

#[test]
fn render_plan_validation_rejects_layout_and_contract_errors() {
    let (_, report) = common::phase_05_report_cases()
        .into_iter()
        .find(|(name, _)| name == "returns")
        .unwrap();
    let base = plan_report(&report, &RenderOptions::default()).unwrap();

    let mut duplicate = base.clone();
    duplicate.panels[1].id = duplicate.panels[0].id.clone();
    assert_invalid_contract(
        duplicate.validate(&report).unwrap_err(),
        "duplicate panel id",
    );

    let mut missing_row = base.clone();
    missing_row.panels[1].row = 3;
    assert_invalid_contract(
        missing_row.validate(&report).unwrap_err(),
        "missing layout row",
    );

    let mut invalid_column = base.clone();
    invalid_column.panels[0].column = 5;
    assert_invalid_contract(
        invalid_column.validate(&report).unwrap_err(),
        "invalid panel column",
    );

    let mut overlap = base.clone();
    overlap.panels[4].row = overlap.panels[3].row;
    overlap.panels[4].column = overlap.panels[3].column;
    overlap.panels[4].column_span = overlap.panels[3].column_span;
    assert_invalid_contract(overlap.validate(&report).unwrap_err(), "panel row overlap");

    let mut unequal_heights = base.clone();
    unequal_heights.panels[4].height += 1;
    assert_invalid_contract(
        unequal_heights.validate(&report).unwrap_err(),
        "unequal panel heights in row",
    );

    let mut too_many = base.clone();
    let first = too_many.panels[0].clone();
    too_many.panels = (0..257)
        .map(|index| PanelPlan {
            id: format!("panel.{index}"),
            row: index as u32,
            ..first.clone()
        })
        .collect();
    assert!(matches!(
        too_many.validate(&report),
        Err(RenderError::LayoutOverflow)
    ));

    let mut too_tall = base.clone();
    too_tall.height = 32_769;
    assert!(matches!(
        too_tall.validate(&report),
        Err(RenderError::LayoutOverflow)
    ));

    let mut missing_table = base.clone();
    missing_table.panels[0].table_ids = vec!["missing.table".to_string()];
    assert!(matches!(
        missing_table.validate(&report),
        Err(RenderError::MissingTable { id }) if id == "missing.table"
    ));
}

#[test]
fn eleven_case_catalog_has_exact_panel_order() {
    let expected = [
        (
            "summary",
            vec![
                "summary.quantile-statistics",
                "summary.returns-summary",
                "summary.quantile-returns",
                "summary.information-summary",
                "summary.turnover-summary",
            ],
        ),
        (
            "returns",
            vec![
                "returns.summary",
                "returns.quantile-bar",
                "returns.daily-distribution",
                "returns.factor-cumulative",
                "returns.quantile-cumulative",
                "returns.mean-spread",
                "returns.group-quantile-bar",
            ],
        ),
        (
            "returns_without_group",
            vec![
                "returns.summary",
                "returns.quantile-bar",
                "returns.daily-distribution",
                "returns.factor-cumulative",
                "returns.quantile-cumulative",
                "returns.mean-spread",
            ],
        ),
        (
            "returns_without_1d_cumulative",
            vec![
                "returns.summary",
                "returns.quantile-bar",
                "returns.daily-distribution",
                "returns.mean-spread",
                "returns.group-quantile-bar",
            ],
        ),
        (
            "information",
            vec![
                "information.summary",
                "information.ic-time-series",
                "information.ic-histogram",
                "information.ic-qq",
                "information.ic-monthly",
                "information.ic-by-group",
            ],
        ),
        (
            "information_without_group",
            vec![
                "information.summary",
                "information.ic-time-series",
                "information.ic-histogram",
                "information.ic-qq",
                "information.ic-monthly",
            ],
        ),
        (
            "turnover",
            vec![
                "turnover.summary",
                "turnover.quantile-series.0",
                "turnover.rank-autocorrelation.0",
                "turnover.quantile-series.1",
                "turnover.rank-autocorrelation.1",
            ],
        ),
        (
            "event_returns",
            vec!["events.average-cumulative", "events.coverage"],
        ),
        (
            "event_study",
            vec![
                "events.quantile-statistics",
                "events.distribution",
                "events.average-cumulative",
                "events.coverage",
                "events.quantile-bar",
                "events.daily-distribution",
            ],
        ),
        (
            "event_study_without_event_returns",
            vec![
                "events.quantile-statistics",
                "events.distribution",
                "events.quantile-bar",
                "events.daily-distribution",
            ],
        ),
    ];

    let cases = common::phase_05_report_cases();
    for (name, ids) in expected {
        let report = cases
            .iter()
            .find(|(case_name, _)| case_name == name)
            .unwrap()
            .1
            .clone();
        let plan = plan_report(&report, &RenderOptions::default()).unwrap();
        assert_eq!(panel_ids(&plan), ids, "{name}");
        assert_unique_table_panels(&plan);
    }

    let full = cases
        .iter()
        .find(|(case_name, _)| case_name == "full")
        .unwrap()
        .1
        .clone();
    let full_ids = panel_ids(&plan_report(&full, &RenderOptions::default()).unwrap());
    assert!(full_ids.starts_with(&[
        "full.quantile-statistics".to_string(),
        "full.returns-summary".to_string(),
        "full.quantile-bar".to_string(),
    ]));
    assert!(full_ids.contains(&"full.group-quantile-bar".to_string()));
    assert!(full_ids.contains(&"full.ic-by-group".to_string()));
}

#[test]
fn responsive_layout_uses_contiguous_non_overlapping_rows() {
    let (_, report) = common::phase_05_report_cases()
        .into_iter()
        .find(|(name, _)| name == "full")
        .unwrap();

    for width in [720, 959, 960, 1200, 4096] {
        let plan = plan_report(
            &report,
            &RenderOptions {
                width,
                ..RenderOptions::default()
            },
        )
        .unwrap();
        assert_eq!(plan.width, width);
        assert_contiguous_rows(&plan);
        assert_rows_do_not_overlap(&plan);
        assert_eq!(plan.height, expected_height(&plan));
        assert!(plot_body_is_large_enough(width, &plan));
        if width < 960 {
            assert!(plan.panels.iter().all(|panel| panel.column_span == 12));
        } else {
            assert!(plan.panels.iter().any(|panel| panel.column_span == 6));
        }
    }
}

fn panel_ids(plan: &RenderPlan) -> Vec<String> {
    plan.panels.iter().map(|panel| panel.id.clone()).collect()
}

fn assert_invalid_contract(error: RenderError, reason: &str) {
    match error {
        RenderError::InvalidContract { reason: actual } => assert_eq!(actual, reason),
        other => panic!("expected InvalidContract, got {other:?}"),
    }
}

fn assert_unique_table_panels(plan: &RenderPlan) {
    let mut seen = HashSet::new();
    for panel in &plan.panels {
        assert!(seen.insert(panel.id.as_str()));
    }
}

fn assert_contiguous_rows(plan: &RenderPlan) {
    let rows = plan
        .panels
        .iter()
        .map(|panel| panel.row)
        .collect::<HashSet<_>>();
    let max = rows.iter().copied().max().unwrap_or(0);
    assert_eq!(rows.len(), max as usize + 1);
}

fn assert_rows_do_not_overlap(plan: &RenderPlan) {
    for left in &plan.panels {
        for right in &plan.panels {
            if left.id >= right.id || left.row != right.row {
                continue;
            }
            assert!(
                left.column + left.column_span <= right.column
                    || right.column + right.column_span <= left.column
            );
        }
    }
}

fn expected_height(plan: &RenderPlan) -> u32 {
    let max_row = plan.panels.iter().map(|panel| panel.row).max().unwrap();
    let mut height = 24 + 56 + 24;
    for row in 0..=max_row {
        if row > 0 {
            height += 24;
        }
        height += plan
            .panels
            .iter()
            .find(|panel| panel.row == row)
            .unwrap()
            .height;
    }
    height
}

fn plot_body_is_large_enough(width: u32, plan: &RenderPlan) -> bool {
    plan.panels.iter().all(|panel| {
        let column_gap = if panel.column_span == 12 { 0 } else { 16 };
        let panel_width = ((width - 48 - column_gap) * u32::from(panel.column_span)) / 12;
        panel_width >= 160 && panel.height >= 120
    })
}
