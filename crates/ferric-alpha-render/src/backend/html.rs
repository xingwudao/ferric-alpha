use std::path::Path;

use ferric_alpha::{ReportTable, TearSheetData};

use crate::charts::common::escape_xml;
use crate::{
    PanelKind, PanelPlan, RenderError, RenderOptions, RenderResult, format_cell, plan_report,
    render_panel_svg,
};

pub fn render_html(report: &TearSheetData, options: &RenderOptions) -> RenderResult<String> {
    let fragment = render_html_fragment(report, options)?;
    Ok(format!(
        "<!doctype html><html><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>Ferric Alpha Report</title><style>{}</style></head><body>{}</body></html>",
        css(),
        fragment
    ))
}

pub fn render_html_fragment(
    report: &TearSheetData,
    options: &RenderOptions,
) -> RenderResult<String> {
    let plan = plan_report(report, options)?;
    let theme = crate::theme_for(options.theme);
    let mut out = format!(
        "<div class=\"ferric-alpha-report\" data-report-kind=\"{:?}\" style=\"max-width:{}px\">",
        plan.report_kind, plan.width
    );
    out.push_str(&format!("<h1>{}</h1>", escape_xml(&plan.title)));
    out.push_str("<div class=\"ferric-alpha-grid\">");
    for panel in &plan.panels {
        out.push_str(&format!(
            "<section class=\"ferric-alpha-panel ferric-alpha-span-{}\" id=\"html-{}\"><h2>{}</h2>",
            panel.column_span,
            escape_xml(&panel.id),
            escape_xml(&panel.title)
        ));
        if matches!(panel.kind, PanelKind::Table) {
            for table_id in &panel.table_ids {
                let table = report
                    .table(table_id)
                    .ok_or_else(|| RenderError::MissingTable {
                        id: table_id.clone(),
                    })?;
                out.push_str(&table_html(table)?);
            }
        } else {
            let rendered = render_panel_svg(report, panel, panel_width(plan.width, panel), theme)?;
            out.push_str(&rendered.svg);
        }
        out.push_str("</section>");
    }
    out.push_str("</div></div>");
    Ok(out)
}

pub fn write_html(
    report: &TearSheetData,
    options: &RenderOptions,
    path: impl AsRef<Path>,
) -> RenderResult<()> {
    let path = path.as_ref();
    let html = render_html(report, options)?;
    std::fs::write(path, html).map_err(|source| RenderError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn table_html(table: &ReportTable) -> RenderResult<String> {
    let mut out = String::new();
    out.push_str("<div class=\"ferric-alpha-table-wrap\"><table><thead><tr>");
    for column in table.columns() {
        out.push_str(&format!("<th>{}</th>", escape_xml(column.label())));
    }
    out.push_str("</tr></thead><tbody>");
    for row in 0..table.row_count() {
        out.push_str("<tr>");
        for column in table.columns() {
            let value = format_cell(column.data(), row)?;
            out.push_str(&format!("<td>{}</td>", escape_xml(&value)));
        }
        out.push_str("</tr>");
    }
    if table.row_count() == 0 {
        out.push_str("<tr><td>No rows</td></tr>");
    }
    out.push_str("</tbody></table></div>");
    Ok(out)
}

fn css() -> &'static str {
    ".ferric-alpha-report{font-family:system-ui,-apple-system,BlinkMacSystemFont,\"Segoe UI\",sans-serif;color:#1f2937;background:#fff}.ferric-alpha-report h1{font-size:22px}.ferric-alpha-grid{display:grid;grid-template-columns:repeat(12,minmax(0,1fr));gap:24px}.ferric-alpha-panel{overflow:hidden}.ferric-alpha-span-12{grid-column:span 12}.ferric-alpha-span-6{grid-column:span 6}.ferric-alpha-panel h2{font-size:16px}.ferric-alpha-table-wrap{overflow-x:auto}.ferric-alpha-table-wrap table{border-collapse:collapse;width:100%;font-size:12px}.ferric-alpha-table-wrap th,.ferric-alpha-table-wrap td{border:1px solid #e5e7eb;padding:4px 6px;white-space:nowrap}@media(max-width:960px){.ferric-alpha-grid{display:block}.ferric-alpha-panel{margin-bottom:24px}}"
}

fn panel_width(plan_width: u32, panel: &PanelPlan) -> u32 {
    let content = plan_width - 48;
    if panel.column_span == 12 {
        content
    } else {
        (content - 16) / 2
    }
}
