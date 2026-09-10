use std::collections::BTreeMap;

use ferric_alpha::TearSheetData;

use crate::charts::common::{
    Rect, TextAllocation, checked_padded_range, checked_visual, escape_xml,
};
use crate::data::{datetimes, float64, int32, require_table, strings, uint32, uint64};
use crate::{PanelKind, PanelPlan, RenderError, RenderResult, Theme};

const PLOT_LEFT: u32 = 64;
const PLOT_RIGHT: u32 = 24;
const PLOT_TOP: u32 = 48;
const PLOT_BOTTOM: u32 = 46;

pub fn render_event_panel_svg(
    report: &TearSheetData,
    panel: &PanelPlan,
    width: u32,
    theme: Theme,
) -> RenderResult<(String, Vec<TextAllocation>)> {
    match panel.kind {
        PanelKind::EventDistribution => render_distribution(report, panel, width, theme),
        PanelKind::EventAverageCumulative { by_group } => {
            render_average_cumulative(report, panel, width, theme, by_group)
        }
        _ => Err(RenderError::InvalidContract {
            reason: "panel kind is not an event chart".to_string(),
        }),
    }
}

fn render_distribution(
    report: &TearSheetData,
    panel: &PanelPlan,
    width: u32,
    theme: Theme,
) -> RenderResult<(String, Vec<TextAllocation>)> {
    let table = require_table(report, &panel.table_ids[0])?;
    let starts = datetimes(table, "bin_start")?;
    let ends = datetimes(table, "bin_end")?;
    let counts = uint64(table, "event_count")?;
    let mut previous_end = None;
    for row in 0..table.row_count() {
        let Some(start) = starts.values[row] else {
            continue;
        };
        let Some(end) = ends.values[row] else {
            continue;
        };
        if start >= end || previous_end.is_some_and(|previous| start < previous) {
            return Err(RenderError::InvalidContract {
                reason: "event distribution bins must be increasing and non-overlapping"
                    .to_string(),
            });
        }
        previous_end = Some(end);
    }
    let max_count = counts.iter().copied().flatten().max().unwrap_or(1) as f64;
    let rect = plot_rect(width, panel.height);
    let mut marks = axis_svg(rect, (0.0, max_count), &theme);
    let lane_width = rect.width as f64 / table.row_count().max(1) as f64;
    for (row, count) in counts.iter().enumerate().take(table.row_count()) {
        let Some(count) = *count else { continue };
        let x = rect.x as f64 + row as f64 * lane_width + lane_width * 0.1;
        let y = scale_y(count as f64, (0.0, max_count), rect);
        marks.push_str(&format!(
            r#"<rect data-mark="event-bin" data-bin="{}" x="{:.3}" y="{:.3}" width="{:.3}" height="{:.3}" fill="{}"/>"#,
            row,
            x,
            y,
            (lane_width * 0.8).max(1.0),
            rect.y as f64 + rect.height as f64 - y,
            theme.accent.hex()
        ));
    }
    Ok((chart_svg(panel, width, theme, &marks), Vec::new()))
}

fn render_average_cumulative(
    report: &TearSheetData,
    panel: &PanelPlan,
    width: u32,
    theme: Theme,
    by_group: bool,
) -> RenderResult<(String, Vec<TextAllocation>)> {
    let table = require_table(report, &panel.table_ids[0])?;
    let quantiles = uint32(table, "factor_quantile")?;
    let offsets = int32(table, "offset")?;
    let means = float64(table, "mean_cumulative_return")?;
    let stds = float64(table, "std_cumulative_return").unwrap_or(&[]);
    let groups = if by_group {
        Some(strings(table, "group")?)
    } else {
        None
    };
    let mut range_values = Vec::new();
    for (row, mean_value) in means.iter().enumerate().take(table.row_count()) {
        range_values.push(*mean_value);
        if let (Some(mean), Some(std)) = (*mean_value, stds.get(row).copied().flatten()) {
            range_values.push(Some(checked_visual(mean + std, "event uncertainty")?));
            range_values.push(Some(checked_visual(mean - std, "event uncertainty")?));
        }
    }
    let y_range = checked_padded_range(range_values.into_iter(), true)?.unwrap_or((-1.0, 1.0));
    let x_min = offsets.iter().copied().flatten().min().unwrap_or(0);
    let x_max = offsets.iter().copied().flatten().max().unwrap_or(0);
    let rect = plot_rect(width, panel.height);
    let mut marks = axis_svg(rect, y_range, &theme);
    let zero_x = scale_offset_x(0, (x_min, x_max), rect);
    marks.push_str(&format!(
        r#"<line data-role="event-zero-line" x1="{:.3}" y1="{}" x2="{:.3}" y2="{}" stroke="{}" stroke-width="1"/>"#,
        zero_x,
        rect.y,
        zero_x,
        rect.y + rect.height,
        theme.grid.hex()
    ));
    let mut mean_series = BTreeMap::<String, Vec<(i32, Option<f64>)>>::new();
    let mut upper_series = BTreeMap::<String, Vec<(i32, Option<f64>)>>::new();
    let mut lower_series = BTreeMap::<String, Vec<(i32, Option<f64>)>>::new();
    for row in 0..table.row_count() {
        let Some(offset) = offsets[row] else { continue };
        let quantile = quantiles[row]
            .map(|value| value.to_string())
            .unwrap_or_default();
        let key = groups
            .and_then(|values| values[row].as_deref())
            .map_or_else(
                || format!("Q{quantile}"),
                |group| format!("{group}:Q{quantile}"),
            );
        mean_series
            .entry(key.clone())
            .or_default()
            .push((offset, means[row]));
        let band = means[row]
            .zip(stds.get(row).copied().flatten())
            .map(|(mean, std)| {
                Ok::<_, RenderError>((
                    checked_visual(mean + std, "event uncertainty")?,
                    checked_visual(mean - std, "event uncertainty")?,
                ))
            })
            .transpose()?;
        upper_series
            .entry(key.clone())
            .or_default()
            .push((offset, band.map(|value| value.0)));
        lower_series
            .entry(key)
            .or_default()
            .push((offset, band.map(|value| value.1)));
    }
    for (key, points) in mean_series {
        let color = Theme::series_color_for_key(&key).hex();
        if let (Some(upper), Some(lower)) = (upper_series.get(&key), lower_series.get(&key)) {
            marks.push_str(&offset_paths(
                "band-upper",
                &key,
                upper,
                (x_min, x_max),
                y_range,
                rect,
                &color,
            )?);
            marks.push_str(&offset_paths(
                "band-lower",
                &key,
                lower,
                (x_min, x_max),
                y_range,
                rect,
                &color,
            )?);
        }
        marks.push_str(&offset_paths(
            "line",
            &key,
            &points,
            (x_min, x_max),
            y_range,
            rect,
            &color,
        )?);
    }
    Ok((chart_svg(panel, width, theme, &marks), Vec::new()))
}

fn offset_paths(
    mark: &str,
    key: &str,
    points: &[(i32, Option<f64>)],
    x_range: (i32, i32),
    y_range: (f64, f64),
    rect: Rect,
    color: &str,
) -> RenderResult<String> {
    let mut out = String::new();
    let mut current = Vec::new();
    for (offset, value) in points {
        if let Some(value) = value {
            checked_visual(*value, "event path")?;
            current.push((
                scale_offset_x(*offset, x_range, rect),
                scale_y(*value, y_range, rect),
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
    let opacity = if mark.starts_with("band") {
        "0.18"
    } else {
        "1"
    };
    out.push_str(&format!(
        r#"<path data-mark="{mark}" data-series="{}" d="{}" fill="none" stroke="{}" stroke-opacity="{}" stroke-width="2"/>"#,
        escape_xml(key),
        data,
        color,
        opacity
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

fn scale_offset_x(value: i32, range: (i32, i32), rect: Rect) -> f64 {
    let ratio = if range.0 == range.1 {
        0.5
    } else {
        f64::from(value - range.0) / f64::from(range.1 - range.0)
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
