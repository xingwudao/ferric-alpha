use std::collections::{BTreeMap, BTreeSet};

use ferric_alpha::TearSheetData;

use crate::charts::common::{Rect, TextAllocation, checked_visual, escape_xml};
use crate::data::{float64, require_table, strings, uint32};
use crate::{PanelKind, PanelPlan, RenderError, RenderResult, Theme};

const PLOT_LEFT: u32 = 64;
const PLOT_RIGHT: u32 = 24;
const PLOT_TOP: u32 = 48;
const PLOT_BOTTOM: u32 = 46;

pub fn render_turnover_panel_svg(
    report: &TearSheetData,
    panel: &PanelPlan,
    width: u32,
    theme: Theme,
) -> RenderResult<(String, Vec<TextAllocation>)> {
    match &panel.kind {
        PanelKind::QuantileTurnover { period } => {
            render_quantile_turnover(report, panel, width, theme, period)
        }
        PanelKind::RankAutocorrelation { period } => {
            render_rank_autocorrelation(report, panel, width, theme, period)
        }
        _ => Err(RenderError::InvalidContract {
            reason: "panel kind is not a turnover chart".to_string(),
        }),
    }
}

fn render_quantile_turnover(
    report: &TearSheetData,
    panel: &PanelPlan,
    width: u32,
    theme: Theme,
    period_filter: &str,
) -> RenderResult<(String, Vec<TextAllocation>)> {
    let table = require_table(report, &panel.table_ids[0])?;
    let quantiles = uint32(table, "factor_quantile")?;
    let periods = strings(table, "period")?;
    let values = float64(table, "turnover")?;
    validate_bounds(values, 0.0, 1.0, "turnover")?;
    let present_quantiles = quantiles.iter().copied().flatten().collect::<BTreeSet<_>>();
    let selected = match (
        present_quantiles.iter().next().copied(),
        present_quantiles.iter().next_back().copied(),
    ) {
        (Some(low), Some(high)) if low == high => vec![low],
        (Some(low), Some(high)) => vec![low, high],
        _ => Vec::new(),
    };
    let mut series = BTreeMap::<String, Vec<(usize, Option<f64>)>>::new();
    for row in 0..table.row_count() {
        if periods[row].as_deref() != Some(period_filter) {
            continue;
        }
        let Some(quantile) = quantiles[row] else {
            continue;
        };
        if selected.contains(&quantile) {
            series
                .entry(format!("Q{quantile}"))
                .or_default()
                .push((row, values[row]));
        }
    }
    let rect = plot_rect(width, panel.height);
    let mut marks = axis_svg(rect, (0.0, 1.0), &theme);
    for (key, points) in series {
        marks.push_str(&line_paths(
            "line",
            &key,
            &points,
            table.row_count(),
            (0.0, 1.0),
            rect,
            &Theme::series_color_for_key(&key).hex(),
        )?);
    }
    Ok((chart_svg(panel, width, theme, &marks), Vec::new()))
}

fn render_rank_autocorrelation(
    report: &TearSheetData,
    panel: &PanelPlan,
    width: u32,
    theme: Theme,
    period_filter: &str,
) -> RenderResult<(String, Vec<TextAllocation>)> {
    let table = require_table(report, &panel.table_ids[0])?;
    let periods = strings(table, "period")?;
    let values = float64(table, "autocorrelation")?;
    validate_bounds(values, -1.0, 1.0, "rank autocorrelation")?;
    let mut points = Vec::new();
    for row in 0..table.row_count() {
        if periods[row].as_deref() == Some(period_filter) {
            points.push((row, values[row]));
        }
    }
    let rect = plot_rect(width, panel.height);
    let mut marks = axis_svg(rect, (-1.0, 1.0), &theme);
    marks.push_str(&line_paths(
        "line",
        period_filter,
        &points,
        table.row_count(),
        (-1.0, 1.0),
        rect,
        &Theme::series_color_for_key(period_filter).hex(),
    )?);
    Ok((chart_svg(panel, width, theme, &marks), Vec::new()))
}

fn validate_bounds(
    values: &[Option<f64>],
    low: f64,
    high: f64,
    operation: &'static str,
) -> RenderResult<()> {
    for value in values.iter().copied().flatten() {
        checked_visual(value, operation)?;
        if value < low || value > high {
            return Err(RenderError::InvalidContract {
                reason: format!("{operation} outside [{low}, {high}]"),
            });
        }
    }
    Ok(())
}

fn line_paths(
    mark: &str,
    key: &str,
    points: &[(usize, Option<f64>)],
    total_rows: usize,
    range: (f64, f64),
    rect: Rect,
    color: &str,
) -> RenderResult<String> {
    let mut out = String::new();
    let mut current = Vec::new();
    for (index, value) in points {
        if let Some(value) = value {
            checked_visual(*value, "turnover line")?;
            current.push((
                scale_x(*index, total_rows, rect),
                scale_y(*value, range, rect),
            ));
        } else {
            flush_path(&mut out, mark, key, &current, color);
            current.clear();
        }
    }
    flush_path(&mut out, mark, key, &current, color);
    Ok(out)
}

fn flush_path(out: &mut String, mark: &str, key: &str, points: &[(f64, f64)], color: &str) {
    if points.is_empty() {
        return;
    }
    let data = points
        .iter()
        .enumerate()
        .map(|(index, (x, y))| {
            if index == 0 {
                format!("M {:.3} {:.3}", x, y)
            } else {
                format!("L {:.3} {:.3}", x, y)
            }
        })
        .collect::<Vec<_>>()
        .join(" ");
    out.push_str(&format!(
        r#"<path data-mark="{mark}" data-series="{}" d="{}" fill="none" stroke="{}" stroke-width="2"/>"#,
        escape_xml(key),
        data,
        color
    ));
}

fn chart_svg(panel: &PanelPlan, width: u32, theme: Theme, marks: &str) -> String {
    format!(
        r#"<svg id="{}" xmlns="http://www.w3.org/2000/svg" width="{}" height="{}" viewBox="0 0 {} {}"><rect x="0" y="0" width="{}" height="{}" fill="{}"/><text data-role="title" x="16" y="26" fill="{}" font-size="16">{}</text>{}</svg>"#,
        escape_xml(&panel.id),
        width,
        panel.height,
        width,
        panel.height,
        width,
        panel.height,
        theme.surface.hex(),
        theme.text.hex(),
        escape_xml(&panel.title),
        marks
    )
}

fn axis_svg(rect: Rect, range: (f64, f64), theme: &Theme) -> String {
    let zero = scale_y(0.0, range, rect);
    format!(
        r#"<rect data-role="plot-body" x="{}" y="{}" width="{}" height="{}" fill="none" stroke="{}"/><line data-role="zero-line" x1="{}" y1="{:.3}" x2="{}" y2="{:.3}" stroke="{}" stroke-width="1"/>"#,
        rect.x,
        rect.y,
        rect.width,
        rect.height,
        theme.grid.hex(),
        rect.x,
        zero,
        rect.x + rect.width,
        zero,
        theme.grid.hex()
    )
}

fn plot_rect(width: u32, height: u32) -> Rect {
    Rect {
        x: PLOT_LEFT,
        y: PLOT_TOP,
        width: width.saturating_sub(PLOT_LEFT + PLOT_RIGHT).max(1),
        height: height.saturating_sub(PLOT_TOP + PLOT_BOTTOM).max(1),
    }
}

fn scale_x(index: usize, total: usize, rect: Rect) -> f64 {
    if total <= 1 {
        return rect.x as f64 + rect.width as f64 / 2.0;
    }
    rect.x as f64 + (index as f64 / (total - 1) as f64) * rect.width as f64
}

fn scale_y(value: f64, range: (f64, f64), rect: Rect) -> f64 {
    let ratio = if range.0 == range.1 {
        0.5
    } else {
        (value - range.0) / (range.1 - range.0)
    };
    rect.y as f64 + (1.0 - ratio.clamp(0.0, 1.0)) * rect.height as f64
}
