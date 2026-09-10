use std::collections::BTreeSet;

use ferric_alpha::{ReportKind, ReportTable, TearSheetData};

use crate::data::strings;
use crate::{RenderOptions, RenderResult};

use super::layout::{CHART_HEIGHT, DISTRIBUTION_HEIGHT};
use super::{PanelKind, RenderPlan, panel};

pub fn plan_report(report: &TearSheetData, options: &RenderOptions) -> RenderResult<RenderPlan> {
    let kind = report.metadata().report_kind();
    let panels = match report {
        TearSheetData::Summary(report) => summary_panels("summary", &report.data),
        TearSheetData::Returns(report) => returns_panels("returns", &report.data),
        TearSheetData::Information(report) => information_panels("information", &report.data),
        TearSheetData::Turnover(report) => turnover_panels("turnover", &report.data)?,
        TearSheetData::Full(report) => {
            let mut panels = vec![panel(
                "full.quantile-statistics",
                "Quantile Statistics",
                6,
                table_height(&report.data.quantile_statistics),
                vec!["quantile.statistics"],
                PanelKind::Table,
            )];
            panels.extend(returns_panels("full", &report.data.returns));
            panels.extend(information_panels("full", &report.data.information));
            panels.extend(turnover_panels("full", &report.data.turnover)?);
            panels
        }
        TearSheetData::EventReturns(report) => {
            vec![
                panel(
                    "events.average-cumulative",
                    "Average Cumulative Return",
                    12,
                    CHART_HEIGHT,
                    vec!["events.average_cumulative_returns"],
                    PanelKind::EventAverageCumulative {
                        by_group: report.options.by_group,
                    },
                ),
                panel(
                    "events.coverage",
                    "Event Coverage",
                    12,
                    table_height(&report.data.coverage),
                    vec!["events.coverage"],
                    PanelKind::Table,
                ),
            ]
        }
        TearSheetData::EventStudy(report) => {
            let mut panels = vec![
                panel(
                    "events.quantile-statistics",
                    "Quantile Statistics",
                    6,
                    table_height(&report.data.quantile_statistics),
                    vec!["quantile.statistics"],
                    PanelKind::Table,
                ),
                panel(
                    "events.distribution",
                    "Events Distribution",
                    6,
                    DISTRIBUTION_HEIGHT,
                    vec!["events.distribution"],
                    PanelKind::EventDistribution,
                ),
            ];
            if report.data.event_returns.is_some() && report.options.event_window.is_some() {
                panels.push(panel(
                    "events.average-cumulative",
                    "Average Cumulative Return",
                    12,
                    CHART_HEIGHT,
                    vec!["events.average_cumulative_returns"],
                    PanelKind::EventAverageCumulative { by_group: false },
                ));
                panels.push(panel(
                    "events.coverage",
                    "Event Coverage",
                    12,
                    table_height(&report.data.event_returns.as_ref().unwrap().coverage),
                    vec!["events.coverage"],
                    PanelKind::Table,
                ));
            }
            panels.extend(event_returns_style_panels());
            panels
        }
    };

    let plan = RenderPlan::new(kind, title_for(kind), options, panels)?;
    plan.validate(report)?;
    Ok(plan)
}

fn summary_panels(prefix: &str, data: &ferric_alpha::SummaryReportData) -> Vec<super::PanelPlan> {
    vec![
        panel(
            "summary.quantile-statistics",
            "Quantile Statistics",
            6,
            table_group_height(&[&data.quantile_statistics]),
            vec!["quantile.statistics"],
            PanelKind::Table,
        ),
        panel(
            format!("{prefix}.returns-summary"),
            "Returns Summary",
            6,
            table_group_height(&[&data.returns_summary, &data.returns_mean_by_quantile]),
            vec!["returns.summary", "returns.mean_by_quantile"],
            PanelKind::Table,
        ),
        panel(
            format!("{prefix}.quantile-returns"),
            "Mean Return by Quantile",
            12,
            CHART_HEIGHT,
            vec!["returns.mean_by_quantile"],
            PanelKind::QuantileReturnsBar { by_group: false },
        ),
        panel(
            format!("{prefix}.information-summary"),
            "Information Summary",
            6,
            table_group_height(&[&data.information_summary]),
            vec!["information.summary"],
            PanelKind::Table,
        ),
        panel(
            format!("{prefix}.turnover-summary"),
            "Turnover Summary",
            6,
            table_group_height(&[
                &data.turnover_mean_by_quantile,
                &data.turnover_mean_rank_autocorrelation,
            ]),
            vec![
                "turnover.mean_by_quantile",
                "turnover.mean_rank_autocorrelation",
            ],
            PanelKind::Table,
        ),
    ]
}

fn returns_panels(prefix: &str, data: &ferric_alpha::ReturnsReportData) -> Vec<super::PanelPlan> {
    let mut panels = vec![
        panel(
            format!("{prefix}.returns-summary"),
            "Returns Summary",
            12,
            table_group_height(&[&data.summary, &data.mean_by_quantile]),
            vec!["returns.summary", "returns.mean_by_quantile"],
            PanelKind::Table,
        ),
        panel(
            format!("{prefix}.quantile-bar"),
            "Mean Return by Quantile",
            6,
            CHART_HEIGHT,
            vec!["returns.mean_by_quantile"],
            PanelKind::QuantileReturnsBar { by_group: false },
        ),
        panel(
            format!("{prefix}.daily-distribution"),
            "Daily Return Distribution by Quantile",
            6,
            DISTRIBUTION_HEIGHT,
            vec!["returns.daily_by_quantile"],
            PanelKind::QuantileReturnsDistribution {
                samples: data.daily_by_quantile.row_count() as u32,
            },
        ),
    ];
    if data.factor_cumulative_1d.is_some() {
        panels.push(panel(
            format!("{prefix}.factor-cumulative"),
            "Factor Cumulative Return",
            6,
            CHART_HEIGHT,
            vec!["returns.factor_cumulative_1d"],
            PanelKind::CumulativeReturns { by_quantile: false },
        ));
    }
    if data.quantile_cumulative_1d.is_some() {
        panels.push(panel(
            format!("{prefix}.quantile-cumulative"),
            "Quantile Cumulative Return",
            6,
            CHART_HEIGHT,
            vec!["returns.quantile_cumulative_1d"],
            PanelKind::CumulativeReturns { by_quantile: true },
        ));
    }
    panels.push(panel(
        format!("{prefix}.mean-spread"),
        "Mean Quantile Return Spread",
        12,
        CHART_HEIGHT,
        vec!["returns.daily_spread"],
        PanelKind::MeanSpread,
    ));
    if data.group_mean_by_quantile.is_some() {
        panels.push(panel(
            format!("{prefix}.group-quantile-bar"),
            "Mean Return by Quantile and Group",
            12,
            CHART_HEIGHT,
            vec!["returns.group_mean_by_quantile"],
            PanelKind::QuantileReturnsBar { by_group: true },
        ));
    }
    rename_returns_summary(prefix, panels)
}

fn rename_returns_summary(
    prefix: &str,
    mut panels: Vec<super::PanelPlan>,
) -> Vec<super::PanelPlan> {
    if prefix == "returns" {
        panels[0].id = "returns.summary".to_string();
    }
    panels
}

fn information_panels(
    prefix: &str,
    data: &ferric_alpha::InformationReportData,
) -> Vec<super::PanelPlan> {
    let first_id = if prefix == "information" {
        "information.summary".to_string()
    } else {
        format!("{prefix}.information-summary")
    };
    let mut panels = vec![
        panel(
            first_id,
            "Information Summary",
            12,
            table_height(&data.summary),
            vec!["information.summary"],
            PanelKind::Table,
        ),
        panel(
            format!("{prefix}.ic-time-series"),
            "Information Coefficient",
            12,
            CHART_HEIGHT,
            vec!["information.ic_by_date", "information.ic_rolling"],
            PanelKind::IcTimeSeries,
        ),
        panel(
            format!("{prefix}.ic-histogram"),
            "IC Histogram",
            6,
            CHART_HEIGHT,
            vec!["information.ic_by_date"],
            PanelKind::IcHistogram { bins: 20 },
        ),
        panel(
            format!("{prefix}.ic-qq"),
            "IC Normal Q-Q",
            6,
            CHART_HEIGHT,
            vec!["information.qq_normal"],
            PanelKind::IcQq,
        ),
        panel(
            format!("{prefix}.ic-monthly"),
            "Monthly Mean IC",
            12,
            DISTRIBUTION_HEIGHT,
            vec!["information.ic_monthly"],
            PanelKind::IcMonthlyHeatmap,
        ),
    ];
    if data.ic_by_group.is_some() {
        panels.push(panel(
            format!("{prefix}.ic-by-group"),
            "Mean IC by Group",
            12,
            CHART_HEIGHT,
            vec!["information.ic_by_group"],
            PanelKind::IcByGroup,
        ));
    }
    panels
}

fn turnover_panels(
    prefix: &str,
    data: &ferric_alpha::TurnoverReportData,
) -> RenderResult<Vec<super::PanelPlan>> {
    let summary_id = if prefix == "turnover" {
        "turnover.summary".to_string()
    } else {
        format!("{prefix}.turnover-summary")
    };
    let mut panels = vec![panel(
        summary_id,
        "Turnover Summary",
        12,
        table_group_height(&[&data.mean_by_quantile, &data.mean_rank_autocorrelation]),
        vec![
            "turnover.mean_by_quantile",
            "turnover.mean_rank_autocorrelation",
        ],
        PanelKind::Table,
    )];

    for (index, period) in table_periods(&data.quantile_series)?
        .into_iter()
        .enumerate()
    {
        panels.push(panel(
            format!("{prefix}.quantile-series.{index}"),
            format!("Quantile Turnover {period}"),
            6,
            CHART_HEIGHT,
            vec!["turnover.quantile_series"],
            PanelKind::QuantileTurnover {
                period: period.clone(),
            },
        ));
        panels.push(panel(
            format!("{prefix}.rank-autocorrelation.{index}"),
            format!("Rank Autocorrelation {period}"),
            6,
            CHART_HEIGHT,
            vec!["turnover.rank_autocorrelation_series"],
            PanelKind::RankAutocorrelation { period },
        ));
    }
    Ok(panels)
}

fn event_returns_style_panels() -> Vec<super::PanelPlan> {
    vec![
        panel(
            "events.quantile-bar",
            "Mean Return by Quantile",
            6,
            CHART_HEIGHT,
            vec!["returns.mean_by_quantile"],
            PanelKind::QuantileReturnsBar { by_group: false },
        ),
        panel(
            "events.daily-distribution",
            "Daily Return Distribution by Quantile",
            6,
            DISTRIBUTION_HEIGHT,
            vec!["returns.daily_by_quantile"],
            PanelKind::QuantileReturnsDistribution { samples: 0 },
        ),
    ]
}

fn table_periods(table: &ReportTable) -> RenderResult<Vec<String>> {
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for period in strings(table, "period")?.iter().flatten() {
        if seen.insert(period.clone()) {
            out.push(period.clone());
        }
    }
    Ok(out)
}

fn table_height(table: &ReportTable) -> u32 {
    36 + 30 * (1 + table.row_count().max(1) as u32)
}

fn table_group_height(tables: &[&ReportTable]) -> u32 {
    let mut height = 36;
    for (index, table) in tables.iter().enumerate() {
        if index > 0 {
            height += 10;
        }
        if tables.len() > 1 {
            height += 30;
        }
        height += 30 + 30 * table.row_count().max(1) as u32;
    }
    height
}

fn title_for(kind: ReportKind) -> &'static str {
    match kind {
        ReportKind::Summary => "Summary Tear Sheet",
        ReportKind::Returns => "Returns Tear Sheet",
        ReportKind::Information => "Information Tear Sheet",
        ReportKind::Turnover => "Turnover Tear Sheet",
        ReportKind::Full => "Full Tear Sheet",
        ReportKind::EventReturns => "Event Returns Tear Sheet",
        ReportKind::EventStudy => "Event Study Tear Sheet",
    }
}
