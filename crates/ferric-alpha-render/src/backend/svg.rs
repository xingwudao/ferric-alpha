use std::path::Path;

use ferric_alpha::{ReportTable, TearSheetData};
use plotters::prelude::{IntoDrawingArea, SVGBackend};

use crate::charts::common::TextAllocation;
use crate::charts::events::render_event_panel_svg;
use crate::charts::information::render_information_panel_svg;
use crate::charts::returns::render_returns_panel_svg;
use crate::charts::table::draw_table_panel;
use crate::charts::turnover::render_turnover_panel_svg;
use crate::{PanelKind, PanelPlan, RenderError, RenderOptions, RenderResult, Theme};
use crate::{plan_report, theme_for};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderedPanel {
    pub svg: String,
    pub text_allocations: Vec<TextAllocation>,
}

pub fn render_svg(report: &TearSheetData, options: &RenderOptions) -> RenderResult<String> {
    let plan = plan_report(report, options)?;
    let theme = theme_for(options.theme);
    let mut out = String::new();
    out.push_str(&format!(
        r#"<svg xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}">"#,
        plan.width, plan.height, plan.width, plan.height
    ));
    out.push_str(&format!(
        r#"<rect x="0" y="0" width="{}" height="{}" fill="{}"/>"#,
        plan.width,
        plan.height,
        theme.background.hex()
    ));

    for panel in &plan.panels {
        let panel_width = panel_width(plan.width, panel);
        let rendered = render_panel_svg(report, panel, panel_width, theme)?;
        out.push_str(&format!(
            r#"<g transform="translate({}, {})">{}</g>"#,
            panel_x(plan.width, panel),
            panel_y(&plan.panels, panel),
            rendered.svg
        ));
    }
    out.push_str("</svg>");
    Ok(out)
}

pub fn write_svg(
    report: &TearSheetData,
    options: &RenderOptions,
    path: impl AsRef<Path>,
) -> RenderResult<()> {
    let path = path.as_ref();
    let svg = render_svg(report, options)?;
    std::fs::write(path, svg).map_err(|source| RenderError::Io {
        path: path.to_path_buf(),
        source,
    })
}

pub fn render_panel_svg(
    report: &TearSheetData,
    panel: &PanelPlan,
    width: u32,
    theme: Theme,
) -> RenderResult<RenderedPanel> {
    match panel.kind {
        PanelKind::Table => render_table_panel_svg(report, panel, width, theme),
        PanelKind::QuantileReturnsBar { .. }
        | PanelKind::QuantileReturnsDistribution { .. }
        | PanelKind::CumulativeReturns { .. }
        | PanelKind::MeanSpread => {
            let (svg, text_allocations) = render_returns_panel_svg(report, panel, width, theme)?;
            Ok(RenderedPanel {
                svg,
                text_allocations,
            })
        }
        PanelKind::IcTimeSeries
        | PanelKind::IcHistogram { .. }
        | PanelKind::IcQq
        | PanelKind::IcMonthlyHeatmap
        | PanelKind::IcByGroup => {
            let (svg, text_allocations) =
                render_information_panel_svg(report, panel, width, theme)?;
            Ok(RenderedPanel {
                svg,
                text_allocations,
            })
        }
        PanelKind::QuantileTurnover { .. } | PanelKind::RankAutocorrelation { .. } => {
            let (svg, text_allocations) = render_turnover_panel_svg(report, panel, width, theme)?;
            Ok(RenderedPanel {
                svg,
                text_allocations,
            })
        }
        PanelKind::EventDistribution | PanelKind::EventAverageCumulative { .. } => {
            let (svg, text_allocations) = render_event_panel_svg(report, panel, width, theme)?;
            Ok(RenderedPanel {
                svg,
                text_allocations,
            })
        }
    }
}

pub fn render_table_panel_svg(
    report: &TearSheetData,
    panel: &PanelPlan,
    width: u32,
    theme: Theme,
) -> RenderResult<RenderedPanel> {
    let tables = panel
        .table_ids
        .iter()
        .map(|id| {
            report
                .table(id)
                .ok_or_else(|| RenderError::MissingTable { id: id.to_string() })
        })
        .collect::<RenderResult<Vec<_>>>()?;
    render_table_svg(&panel.id, &panel.title, &tables, width, panel.height, theme)
}

pub fn render_table_svg(
    panel_id: &str,
    title: &str,
    tables: &[&ReportTable],
    width: u32,
    height: u32,
    theme: Theme,
) -> RenderResult<RenderedPanel> {
    let mut svg = String::new();
    {
        let backend = SVGBackend::with_string(&mut svg, (width, height));
        let area = backend.into_drawing_area();
        let mut manifest = Vec::new();
        draw_table_panel(&area, panel_id, title, tables, &theme, &mut manifest)?;
        area.present().map_err(|error| RenderError::Backend {
            format: "SVG",
            message: error.to_string(),
        })?;
        drop(area);
        Ok(RenderedPanel {
            svg: inject_panel_id(svg, panel_id),
            text_allocations: manifest,
        })
    }
}

fn inject_panel_id(svg: String, panel_id: &str) -> String {
    svg.replacen("<svg ", &format!(r#"<svg id="{panel_id}" "#), 1)
}

fn panel_x(plan_width: u32, panel: &PanelPlan) -> u32 {
    let margin = 24;
    let content_width = plan_width - 48;
    if panel.column == 0 {
        margin
    } else {
        margin + content_width / 2 + 8
    }
}

fn panel_width(plan_width: u32, panel: &PanelPlan) -> u32 {
    let content = plan_width - 48;
    if panel.column_span == 12 {
        content
    } else {
        (content - 16) / 2
    }
}

fn panel_y(panels: &[PanelPlan], panel: &PanelPlan) -> u32 {
    let mut y = 24 + 56 + 24;
    for row in 0..panel.row {
        let height = panels
            .iter()
            .find(|candidate| candidate.row == row)
            .map_or(0, |candidate| candidate.height);
        y += height + 24;
    }
    y
}
