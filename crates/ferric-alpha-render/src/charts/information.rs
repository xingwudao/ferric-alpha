use std::collections::BTreeMap;

use ferric_alpha::TearSheetData;

use crate::charts::common::{
    Rect, TextAllocation, checked_padded_range, checked_visual, escape_xml,
};
use crate::data::{float64, int32, require_table, strings};
use crate::{PanelKind, PanelPlan, RenderError, RenderResult, Theme};

const PLOT_LEFT: u32 = 64;
const PLOT_RIGHT: u32 = 24;
const PLOT_TOP: u32 = 48;
const PLOT_BOTTOM: u32 = 46;

pub fn render_information_panel_svg(
    report: &TearSheetData,
    panel: &PanelPlan,
    width: u32,
    theme: Theme,
) -> RenderResult<(String, Vec<TextAllocation>)> {
    match panel.kind {
        PanelKind::IcTimeSeries => render_ic_series(report, panel, width, theme),
        PanelKind::IcHistogram { bins } => render_histogram(report, panel, width, theme, bins),
        PanelKind::IcQq => render_qq(report, panel, width, theme),
        PanelKind::IcMonthlyHeatmap => render_heatmap(report, panel, width, theme),
        PanelKind::IcByGroup => render_group_bars(report, panel, width, theme),
        _ => Err(RenderError::InvalidContract {
            reason: "panel kind is not an information chart".to_string(),
        }),
    }
}

fn render_ic_series(
    report: &TearSheetData,
    panel: &PanelPlan,
    width: u32,
    theme: Theme,
) -> RenderResult<(String, Vec<TextAllocation>)> {
    let raw = require_table(report, "information.ic_by_date")?;
    let rolling = require_table(report, "information.ic_rolling")?;
    let raw_periods = strings(raw, "period")?;
    let raw_values = float64(raw, "ic")?;
    let rolling_periods = strings(rolling, "period")?;
    let rolling_values = float64(rolling, "rolling_mean_ic")?;
    validate_unit_interval(raw_values, -1.0, 1.0, "raw IC")?;
    validate_unit_interval(rolling_values, -1.0, 1.0, "rolling IC")?;

    let rect = plot_rect(width, panel.height);
    let range = (-1.0, 1.0);
    let mut marks = axis_svg(rect, range, &theme);
    marks.push_str(&series_paths(
        "raw-line",
        raw_periods,
        raw_values,
        raw.row_count(),
        range,
        rect,
    )?);
    marks.push_str(&series_paths(
        "rolling-line",
        rolling_periods,
        rolling_values,
        rolling.row_count(),
        range,
        rect,
    )?);
    Ok((chart_svg(panel, width, theme, &marks), Vec::new()))
}

fn render_histogram(
    report: &TearSheetData,
    panel: &PanelPlan,
    width: u32,
    theme: Theme,
    bins: u32,
) -> RenderResult<(String, Vec<TextAllocation>)> {
    let table = require_table(report, &panel.table_ids[0])?;
    let periods = strings(table, "period")?;
    let values = float64(table, "ic")?;
    validate_unit_interval(values, -1.0, 1.0, "histogram IC")?;
    let bins = bins.max(1) as usize;
    let mut counts = BTreeMap::<String, Vec<u32>>::new();
    let mut means = BTreeMap::<String, (f64, u32)>::new();
    for row in 0..table.row_count() {
        let Some(value) = values[row] else { continue };
        let period = periods[row].clone().unwrap_or_default();
        let mut bin = (((value + 1.0) / 2.0) * bins as f64).floor() as usize;
        if bin >= bins {
            bin = bins - 1;
        }
        counts
            .entry(period.clone())
            .or_insert_with(|| vec![0; bins])[bin] += 1;
        let entry = means.entry(period).or_default();
        entry.0 += value;
        entry.1 += 1;
    }
    let max_count = counts
        .values()
        .flat_map(|values| values.iter().copied())
        .max()
        .unwrap_or(1) as f64;
    let rect = plot_rect(width, panel.height);
    let mut marks = axis_svg(rect, (0.0, max_count), &theme);
    let period_count = counts.len().max(1);
    let group_width = rect.width as f64 / bins as f64;
    for (period_index, (period, values)) in counts.iter().enumerate() {
        let bar_width = group_width / period_count as f64;
        let color = Theme::series_color_for_key(period).hex();
        for (bin, count) in values.iter().enumerate() {
            let x = rect.x as f64 + bin as f64 * group_width + period_index as f64 * bar_width;
            let y = scale_y(*count as f64, (0.0, max_count), rect);
            marks.push_str(&format!(
                r#"<rect data-mark="hist-bin" data-period="{}" data-bin="{}" x="{:.3}" y="{:.3}" width="{:.3}" height="{:.3}" fill="{}" fill-opacity="0.7"/>"#,
                escape_xml(period),
                bin,
                x,
                y,
                bar_width.max(1.0),
                rect.y as f64 + rect.height as f64 - y,
                color
            ));
        }
        if let Some((sum, count)) = means.get(period).copied()
            && count > 0
        {
            let mean = sum / f64::from(count);
            let x = rect.x as f64 + ((mean + 1.0) / 2.0) * rect.width as f64;
            marks.push_str(&format!(
                r#"<line data-mark="mean-line" data-period="{}" x1="{:.3}" y1="{}" x2="{:.3}" y2="{}" stroke="{}" stroke-width="1"/>"#,
                escape_xml(period),
                x,
                rect.y,
                x,
                rect.y + rect.height,
                color
            ));
        }
    }
    Ok((chart_svg(panel, width, theme, &marks), Vec::new()))
}

fn render_qq(
    report: &TearSheetData,
    panel: &PanelPlan,
    width: u32,
    theme: Theme,
) -> RenderResult<(String, Vec<TextAllocation>)> {
    let table = require_table(report, &panel.table_ids[0])?;
    let periods = strings(table, "period")?;
    let theoretical = float64(table, "theoretical")?;
    let observed = float64(table, "observed")?;
    let range = checked_padded_range(
        theoretical.iter().copied().chain(observed.iter().copied()),
        true,
    )?
    .unwrap_or((-1.0, 1.0));
    let rect = plot_rect(width, panel.height);
    let mut marks = axis_svg(rect, range, &theme);
    for row in 0..table.row_count() {
        if let (Some(x_value), Some(y_value)) = (theoretical[row], observed[row]) {
            checked_visual(x_value, "qq theoretical")?;
            checked_visual(y_value, "qq observed")?;
            let period = periods[row].as_deref().unwrap_or("");
            marks.push_str(&format!(
                r#"<circle data-mark="qq-point" data-period="{}" cx="{:.3}" cy="{:.3}" r="2.5" fill="{}"/>"#,
                escape_xml(period),
                scale_value_x(x_value, range, rect),
                scale_y(y_value, range, rect),
                Theme::series_color_for_key(period).hex()
            ));
        }
    }
    Ok((chart_svg(panel, width, theme, &marks), Vec::new()))
}

fn render_heatmap(
    report: &TearSheetData,
    panel: &PanelPlan,
    width: u32,
    theme: Theme,
) -> RenderResult<(String, Vec<TextAllocation>)> {
    let table = require_table(report, &panel.table_ids[0])?;
    let months = int32(table, "time_bucket")?
        .iter()
        .map(|value| value.map(|value| value.to_string()))
        .collect::<Vec<_>>();
    let periods = strings(table, "period")?;
    let values = float64(table, "mean_ic")?;
    validate_unit_interval(values, -1.0, 1.0, "monthly IC")?;
    let month_order = ordered_strings(&months);
    let period_order = report.metadata().periods().to_vec();
    let max_abs = values
        .iter()
        .copied()
        .flatten()
        .map(f64::abs)
        .fold(0.0, f64::max)
        .max(1e-12);
    let rect = plot_rect(width, panel.height);
    let cell_w = rect.width as f64 / period_order.len().max(1) as f64;
    let cell_h = rect.height as f64 / month_order.len().max(1) as f64;
    let mut marks = format!(
        r#"<rect data-role="plot-body" x="{}" y="{}" width="{}" height="{}" fill="none" stroke="{}"/>"#,
        rect.x,
        rect.y,
        rect.width,
        rect.height,
        theme.grid.hex()
    );
    for row in 0..table.row_count() {
        let Some(month) = months[row].as_ref() else {
            continue;
        };
        let Some(period) = periods[row].as_ref() else {
            continue;
        };
        let Some(month_index) = month_order.iter().position(|value| value == month) else {
            continue;
        };
        let Some(period_index) = period_order.iter().position(|value| value == period) else {
            continue;
        };
        let Some(value) = values[row] else { continue };
        let intensity = (value.abs() / max_abs).clamp(0.0, 1.0);
        let fill = if value >= 0.0 {
            blend(theme.surface, (22, 163, 74), intensity)
        } else {
            blend(theme.surface, (220, 38, 38), intensity)
        };
        marks.push_str(&format!(
            r#"<rect data-mark="heat-cell" data-month="{}" data-period="{}" x="{:.3}" y="{:.3}" width="{:.3}" height="{:.3}" fill="{}"/>"#,
            escape_xml(month),
            escape_xml(period),
            rect.x as f64 + period_index as f64 * cell_w,
            rect.y as f64 + month_index as f64 * cell_h,
            cell_w,
            cell_h,
            fill
        ));
    }
    Ok((chart_svg(panel, width, theme, &marks), Vec::new()))
}

fn render_group_bars(
    report: &TearSheetData,
    panel: &PanelPlan,
    width: u32,
    theme: Theme,
) -> RenderResult<(String, Vec<TextAllocation>)> {
    let table = require_table(report, &panel.table_ids[0])?;
    let groups = strings(table, "group")?;
    let periods = strings(table, "period")?;
    let values = float64(table, "mean_ic")?;
    validate_unit_interval(values, -1.0, 1.0, "group IC")?;
    let range = checked_padded_range(values.iter().copied(), true)?.unwrap_or((-1.0, 1.0));
    let rect = plot_rect(width, panel.height);
    let mut marks = axis_svg(rect, range, &theme);
    let row_count = table.row_count().max(1);
    let lane_width = rect.width as f64 / row_count as f64;
    for row in 0..table.row_count() {
        let Some(value) = values[row] else { continue };
        let group = groups[row].as_deref().unwrap_or("");
        let period = periods[row].as_deref().unwrap_or("");
        let x = rect.x as f64 + row as f64 * lane_width + lane_width * 0.15;
        let y0 = scale_y(0.0, range, rect);
        let y1 = scale_y(value, range, rect);
        marks.push_str(&format!(
            r#"<rect data-mark="group-bar" data-group="{}" data-period="{}" x="{:.3}" y="{:.3}" width="{:.3}" height="{:.3}" fill="{}"/>"#,
            escape_xml(group),
            escape_xml(period),
            x,
            y0.min(y1),
            (lane_width * 0.7).max(1.0),
            (y0 - y1).abs().max(1.0),
            Theme::series_color_for_key(period).hex()
        ));
    }
    Ok((chart_svg(panel, width, theme, &marks), Vec::new()))
}

fn validate_unit_interval(
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

fn series_paths(
    mark: &str,
    periods: &[Option<String>],
    values: &[Option<f64>],
    total_rows: usize,
    range: (f64, f64),
    rect: Rect,
) -> RenderResult<String> {
    let mut series = BTreeMap::<String, Vec<(usize, Option<f64>)>>::new();
    for (row, value) in values.iter().enumerate().take(total_rows) {
        let key = periods[row].clone().unwrap_or_default();
        series.entry(key).or_default().push((row, *value));
    }
    let mut out = String::new();
    for (period, points) in series {
        out.push_str(&line_paths(
            mark,
            &period,
            &points,
            total_rows,
            range,
            rect,
            &Theme::series_color_for_key(&period).hex(),
        )?);
    }
    Ok(out)
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
            checked_visual(*value, "information line")?;
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

fn scale_value_x(value: f64, range: (f64, f64), rect: Rect) -> f64 {
    let ratio = if range.0 == range.1 {
        0.5
    } else {
        (value - range.0) / (range.1 - range.0)
    };
    rect.x as f64 + ratio.clamp(0.0, 1.0) * rect.width as f64
}

fn scale_y(value: f64, range: (f64, f64), rect: Rect) -> f64 {
    let ratio = if range.0 == range.1 {
        0.5
    } else {
        (value - range.0) / (range.1 - range.0)
    };
    rect.y as f64 + (1.0 - ratio.clamp(0.0, 1.0)) * rect.height as f64
}

fn ordered_strings(values: &[Option<String>]) -> Vec<String> {
    let mut out = Vec::new();
    for value in values.iter().flatten() {
        if !out.contains(value) {
            out.push(value.clone());
        }
    }
    out
}

fn blend(surface: crate::Color, target: (u8, u8, u8), intensity: f64) -> String {
    let mix = |base: u8, target: u8| {
        (f64::from(base) + (f64::from(target) - f64::from(base)) * intensity).round() as u8
    };
    format!(
        "#{:02x}{:02x}{:02x}",
        mix(surface.red, target.0),
        mix(surface.green, target.1),
        mix(surface.blue, target.2)
    )
}
