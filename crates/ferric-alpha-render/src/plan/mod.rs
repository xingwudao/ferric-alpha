mod catalog;
mod layout;

use std::collections::{HashMap, HashSet};

use ferric_alpha::{ReportKind, ReportTable, TearSheetData};
use serde::{Deserialize, Serialize};

use crate::{
    MAX_LOGICAL_HEIGHT, MAX_PANELS, MAX_WIDTH, MIN_WIDTH, RENDER_PLAN_VERSION, RenderError,
    RenderResult,
};
use crate::{RenderOptions, RenderTheme};

pub use catalog::plan_report;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderPlan {
    pub contract_version: String,
    pub report_kind: ReportKind,
    pub title: String,
    pub width: u32,
    pub height: u32,
    pub theme: RenderTheme,
    pub panels: Vec<PanelPlan>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PanelPlan {
    pub id: String,
    pub title: String,
    pub row: u32,
    pub column: u8,
    pub column_span: u8,
    pub height: u32,
    pub table_ids: Vec<String>,
    pub kind: PanelKind,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum PanelKind {
    Table,
    QuantileReturnsBar { by_group: bool },
    QuantileReturnsDistribution { samples: u32 },
    CumulativeReturns { by_quantile: bool },
    MeanSpread,
    IcTimeSeries,
    IcHistogram { bins: u32 },
    IcQq,
    IcMonthlyHeatmap,
    IcByGroup,
    QuantileTurnover { period: String },
    RankAutocorrelation { period: String },
    EventDistribution,
    EventAverageCumulative { by_group: bool },
}

impl RenderPlan {
    pub fn new(
        report_kind: ReportKind,
        title: impl Into<String>,
        options: &RenderOptions,
        panels: Vec<PanelPlan>,
    ) -> RenderResult<Self> {
        validate_plan_options(options)?;
        let panels = layout::assign_layout(options.width, panels)?;
        let height = layout::total_height(&panels)?;
        Ok(Self {
            contract_version: RENDER_PLAN_VERSION.to_string(),
            report_kind,
            title: title.into(),
            width: options.width,
            height,
            theme: options.theme,
            panels,
        })
    }

    pub fn to_json(&self) -> RenderResult<String> {
        serde_json::to_string(self).map_err(|error| RenderError::Backend {
            format: "render-plan",
            message: error.to_string(),
        })
    }

    pub fn validate(&self, report: &TearSheetData) -> RenderResult<()> {
        if self.contract_version != RENDER_PLAN_VERSION {
            return invalid_contract("invalid render-plan contract version");
        }
        if self.report_kind != report.metadata().report_kind() {
            return invalid_contract("report kind mismatch");
        }
        if self.panels.len() > MAX_PANELS || self.height > MAX_LOGICAL_HEIGHT {
            return Err(RenderError::LayoutOverflow);
        }

        let table_map = report
            .table_ids()
            .into_iter()
            .filter_map(|id| report.table(id).map(|table| (id, table)))
            .collect::<HashMap<_, _>>();
        let mut seen_ids = HashSet::with_capacity(self.panels.len());
        let mut rows = HashMap::<u32, Vec<&PanelPlan>>::new();

        for panel in &self.panels {
            if panel.id.is_empty() || !seen_ids.insert(panel.id.as_str()) {
                return invalid_contract("duplicate panel id");
            }
            if panel.column != 0 && panel.column != 6 {
                return invalid_contract("invalid panel column");
            }
            if panel.column_span != 6 && panel.column_span != 12 {
                return invalid_contract("invalid panel span");
            }
            if panel.column + panel.column_span > 12 {
                return invalid_contract("panel row overflow");
            }
            if panel.table_ids.is_empty() {
                return invalid_contract("panel has no table ids");
            }
            for table_id in &panel.table_ids {
                let table = table_map.get(table_id.as_str()).copied().ok_or_else(|| {
                    RenderError::MissingTable {
                        id: table_id.clone(),
                    }
                })?;
                validate_panel_table(panel, table)?;
            }
            rows.entry(panel.row).or_default().push(panel);
        }

        let max_row = rows.keys().copied().max().unwrap_or(0);
        for row in 0..=max_row {
            let panels = rows.get(&row).ok_or_else(|| RenderError::InvalidContract {
                reason: "missing layout row".to_string(),
            })?;
            let first_height = panels[0].height;
            if panels.iter().any(|panel| panel.height != first_height) {
                return invalid_contract("unequal panel heights in row");
            }
            for (index, left) in panels.iter().enumerate() {
                for right in panels.iter().skip(index + 1) {
                    let left_end = left.column + left.column_span;
                    let right_end = right.column + right.column_span;
                    if left.column < right_end && right.column < left_end {
                        return invalid_contract("panel row overlap");
                    }
                }
            }
        }

        Ok(())
    }
}

pub(crate) fn panel(
    id: impl Into<String>,
    title: impl Into<String>,
    column_span: u8,
    height: u32,
    table_ids: Vec<&str>,
    kind: PanelKind,
) -> PanelPlan {
    PanelPlan {
        id: id.into(),
        title: title.into(),
        row: 0,
        column: 0,
        column_span,
        height,
        table_ids: table_ids.into_iter().map(ToOwned::to_owned).collect(),
        kind,
    }
}

fn validate_panel_table(panel: &PanelPlan, table: &ReportTable) -> RenderResult<()> {
    if table.columns().is_empty() {
        return invalid_contract("table panel has no columns");
    }
    match &panel.kind {
        PanelKind::Table => Ok(()),
        PanelKind::QuantileReturnsBar { by_group } => {
            require_columns(table, &["factor_quantile", "period", "mean_return"])?;
            if *by_group {
                require_columns(table, &["group"])?;
            }
            Ok(())
        }
        PanelKind::QuantileReturnsDistribution { .. } => {
            require_columns(table, &["date", "factor_quantile", "period", "mean_return"])
        }
        PanelKind::CumulativeReturns { by_quantile } => {
            let mut columns = vec!["date"];
            if *by_quantile {
                columns.push("factor_quantile");
            }
            columns.push("cumulative_return");
            require_columns(table, &columns)
        }
        PanelKind::MeanSpread => {
            require_columns(table, &["date", "period", "mean_return_difference"])
        }
        PanelKind::IcTimeSeries => require_columns(table, &["period"]),
        PanelKind::IcHistogram { .. } => require_columns(table, &["period", "ic"]),
        PanelKind::IcQq => require_columns(table, &["period", "theoretical", "observed"]),
        PanelKind::IcMonthlyHeatmap => {
            require_columns(table, &["time_bucket", "period", "mean_ic"])
        }
        PanelKind::IcByGroup => require_columns(table, &["group", "period", "mean_ic"]),
        PanelKind::QuantileTurnover { .. } => {
            require_columns(table, &["date", "factor_quantile", "period", "turnover"])
        }
        PanelKind::RankAutocorrelation { .. } => {
            require_columns(table, &["date", "period", "autocorrelation"])
        }
        PanelKind::EventDistribution => require_columns(table, &["bin_start", "bin_end"]),
        PanelKind::EventAverageCumulative { by_group } => {
            let mut columns = vec!["factor_quantile", "offset", "mean_cumulative_return"];
            if *by_group {
                columns.push("group");
            }
            require_columns(table, &columns)
        }
    }
}

fn require_columns(table: &ReportTable, columns: &[&str]) -> RenderResult<()> {
    let available = table.column_names();
    for column in columns {
        if !available.contains(column) {
            return Err(RenderError::MissingColumn {
                table: table.id().to_string(),
                column: (*column).to_string(),
            });
        }
    }
    Ok(())
}

fn invalid_contract<T>(reason: &str) -> RenderResult<T> {
    Err(RenderError::InvalidContract {
        reason: reason.to_string(),
    })
}

fn validate_plan_options(options: &RenderOptions) -> RenderResult<()> {
    if !(MIN_WIDTH..=MAX_WIDTH).contains(&options.width) {
        return Err(RenderError::InvalidOption {
            option: "width",
            reason: "must be in 720..=4096",
        });
    }
    if !options.scale.is_finite() || !(0.5..=4.0).contains(&options.scale) {
        return Err(RenderError::InvalidOption {
            option: "scale",
            reason: "must be finite and in 0.5..=4.0",
        });
    }
    Ok(())
}
