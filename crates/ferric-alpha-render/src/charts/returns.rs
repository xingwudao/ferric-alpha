use std::collections::BTreeMap;

use ferric_alpha::{DisplayUnit, TearSheetData};

use crate::charts::common::{TextAllocation, checked_padded_range, checked_visual, escape_xml};
use crate::data::{datetimes, float64, require_table, strings, uint32};
use crate::{
    PanelKind, PanelPlan, RenderError, RenderResult, Theme, format_datetime, format_number,
};

use super::common::Rect;

const PLOT_LEFT: u32 = 64;
const PLOT_RIGHT: u32 = 24;
const PLOT_TOP: u32 = 48;
const PLOT_BOTTOM: u32 = 46;
const KDE_POINTS: usize = 64;

pub fn render_returns_panel_svg(
    report: &TearSheetData,
    panel: &PanelPlan,
    width: u32,
    theme: Theme,
) -> RenderResult<(String, Vec<TextAllocation>)> {
    match panel.kind {
        PanelKind::QuantileReturnsBar { by_group } => {
            render_quantile_bars(report, panel, width, theme, by_group)
        }
        PanelKind::QuantileReturnsDistribution { .. } => {
            render_distribution(report, panel, width, theme)
        }
        PanelKind::CumulativeReturns { by_quantile } => {
            render_cumulative(report, panel, width, theme, by_quantile)
        }
        PanelKind::MeanSpread => render_spread(report, panel, width, theme),
        _ => Err(RenderError::InvalidContract {
            reason: "panel kind is not a returns chart".to_string(),
        }),
    }
}

fn render_quantile_bars(
    report: &TearSheetData,
    panel: &PanelPlan,
    width: u32,
    theme: Theme,
    by_group: bool,
) -> RenderResult<(String, Vec<TextAllocation>)> {
    let table = require_table(report, &panel.table_ids[0])?;
    let quantiles = uint32(table, "factor_quantile")?;
    let periods = strings(table, "period")?;
    let means = float64(table, "mean_return")?;
    let errors = float64(table, "std_error").unwrap_or(&[]);
    let groups = if by_group {
        Some(strings(table, "group")?)
    } else {
        None
    };
    let range = checked_padded_range(means.iter().copied().chain(errors.iter().copied()), true)?
        .unwrap_or((-1.0, 1.0));
    let rect = plot_rect(width, panel.height);
    let mut marks = String::new();
    marks.push_str(&axis_svg(rect, range, &theme));
    let row_count = table.row_count().max(1);
    let lane_width = rect.width as f64 / row_count as f64;
    for row in 0..table.row_count() {
        let Some(mean) = means[row] else { continue };
        checked_visual(mean, "quantile bar")?;
        let x = rect.x as f64 + row as f64 * lane_width + lane_width * 0.15;
        let bar_width = (lane_width * 0.7).max(1.0);
        let y0 = scale_y(0.0, range, rect);
        let y1 = scale_y(mean, range, rect);
        let y = y0.min(y1);
        let height = (y0 - y1).abs().max(1.0);
        let key = groups
            .and_then(|values| values[row].as_deref())
            .unwrap_or_else(|| periods[row].as_deref().unwrap_or("period"));
        let color = Theme::series_color_for_key(key).hex();
        marks.push_str(&format!(
            r#"<rect data-mark="bar" data-quantile="{}" data-period="{}" x="{:.3}" y="{:.3}" width="{:.3}" height="{:.3}" fill="{}"/>"#,
            quantiles[row].unwrap_or_default(),
            escape_xml(periods[row].as_deref().unwrap_or("")),
            x,
            y,
            bar_width,
            height,
            color
        ));
        if let Some(error) = errors.get(row).copied().flatten() {
            let high = checked_visual(mean + error, "quantile whisker")?;
            let low = checked_visual(mean - error, "quantile whisker")?;
            let x_mid = x + bar_width / 2.0;
            marks.push_str(&format!(
                r#"<line data-mark="whisker" x1="{:.3}" y1="{:.3}" x2="{:.3}" y2="{:.3}" stroke="{}" stroke-width="1"/>"#,
                x_mid,
                scale_y(low, range, rect),
                x_mid,
                scale_y(high, range, rect),
                theme.text.hex()
            ));
        }
    }
    Ok((chart_svg(panel, width, theme, &marks), Vec::new()))
}

fn render_distribution(
    report: &TearSheetData,
    panel: &PanelPlan,
    width: u32,
    theme: Theme,
) -> RenderResult<(String, Vec<TextAllocation>)> {
    let table = require_table(report, &panel.table_ids[0])?;
    let quantiles = uint32(table, "factor_quantile")?;
    let periods = strings(table, "period")?;
    let means = float64(table, "mean_return")?;
    let mut groups = BTreeMap::<(u32, String), Vec<Option<f64>>>::new();
    for row in 0..table.row_count() {
        if let (Some(quantile), Some(period)) = (quantiles[row], periods[row].as_ref()) {
            groups
                .entry((quantile, period.clone()))
                .or_default()
                .push(means[row]);
        }
    }
    let range = checked_padded_range(means.iter().copied(), true)?.unwrap_or((-1.0, 1.0));
    let rect = plot_rect(width, panel.height);
    let mut marks = String::new();
    marks.push_str(&axis_svg(rect, range, &theme));
    let lane_width = rect.width as f64 / groups.len().max(1) as f64;
    for (index, ((quantile, period), values)) in groups.iter().enumerate() {
        let density = kde_density(values)?;
        if density.is_empty() {
            continue;
        }
        let center = rect.x as f64 + (index as f64 + 0.5) * lane_width;
        let half = lane_width * 0.35;
        if density.len() == 1 {
            let y = scale_y(density[0].0, range, rect);
            marks.push_str(&format!(
                r#"<circle data-mark="singleton" data-quantile="{quantile}" data-period="{}" cx="{:.3}" cy="{:.3}" r="3" fill="{}"/>"#,
                escape_xml(period),
                center,
                y,
                Theme::series_color_for_key(period).hex()
            ));
            continue;
        }
        let right = density
            .iter()
            .map(|(x, y)| format!("{:.3},{:.3}", center + half * *y, scale_y(*x, range, rect)))
            .collect::<Vec<_>>();
        let left = density
            .iter()
            .rev()
            .map(|(x, y)| format!("{:.3},{:.3}", center - half * *y, scale_y(*x, range, rect)))
            .collect::<Vec<_>>();
        marks.push_str(&format!(
            r#"<polygon data-mark="violin" data-quantile="{quantile}" data-period="{}" points="{} {}" fill="{}" fill-opacity="0.35" stroke="{}" stroke-width="1"/>"#,
            escape_xml(period),
            right.join(" "),
            left.join(" "),
            Theme::series_color_for_key(period).hex(),
            Theme::series_color_for_key(period).hex()
        ));
    }
    Ok((chart_svg(panel, width, theme, &marks), Vec::new()))
}

fn render_cumulative(
    report: &TearSheetData,
    panel: &PanelPlan,
    width: u32,
    theme: Theme,
    by_quantile: bool,
) -> RenderResult<(String, Vec<TextAllocation>)> {
    let table = require_table(report, &panel.table_ids[0])?;
    let dates = datetimes(table, "date")?;
    let values = float64(table, "cumulative_return")?;
    let quantiles = if by_quantile {
        Some(uint32(table, "factor_quantile")?)
    } else {
        None
    };
    let range = checked_padded_range(values.iter().copied(), true)?.unwrap_or((-1.0, 1.0));
    let rect = plot_rect(width, panel.height);
    let mut marks = String::new();
    marks.push_str(&axis_svg(rect, range, &theme));
    let mut series = BTreeMap::<String, Vec<(usize, Option<f64>)>>::new();
    for (row, value) in values.iter().enumerate().take(table.row_count()) {
        let key = quantiles
            .map(|values| {
                values[row]
                    .map(|value| value.to_string())
                    .unwrap_or_default()
            })
            .unwrap_or_else(|| "factor".to_string());
        series.entry(key).or_default().push((row, *value));
    }
    for (key, points) in series {
        let label = dates
            .values
            .first()
            .copied()
            .flatten()
            .map(|value| format_datetime(value, dates.time_unit, dates.timezone))
            .transpose()?
            .unwrap_or_default();
        marks.push_str(&line_paths(
            "line",
            &key,
            &points,
            table.row_count(),
            range,
            rect,
            &Theme::series_color_for_key(&key).hex(),
        )?);
        marks.push_str(&format!(
            r#"<text data-mark="legend" x="{}" y="{}" fill="{}" font-size="11">{} {}</text>"#,
            rect.x,
            rect.y.saturating_sub(8),
            theme.muted_text.hex(),
            escape_xml(&key),
            escape_xml(&label)
        ));
    }
    Ok((chart_svg(panel, width, theme, &marks), Vec::new()))
}

fn render_spread(
    report: &TearSheetData,
    panel: &PanelPlan,
    width: u32,
    theme: Theme,
) -> RenderResult<(String, Vec<TextAllocation>)> {
    let table = require_table(report, &panel.table_ids[0])?;
    let periods = strings(table, "period")?;
    let means = float64(table, "mean_return_difference")?;
    let errors = float64(table, "joint_std_error").unwrap_or(&[]);
    let mut range_values = Vec::new();
    for (row, mean_value) in means.iter().enumerate().take(table.row_count()) {
        range_values.push(*mean_value);
        if let (Some(mean), Some(error)) = (*mean_value, errors.get(row).copied().flatten()) {
            range_values.push(Some(checked_visual(mean + error, "spread band")?));
            range_values.push(Some(checked_visual(mean - error, "spread band")?));
        }
    }
    let range = checked_padded_range(range_values.into_iter(), true)?.unwrap_or((-1.0, 1.0));
    let rect = plot_rect(width, panel.height);
    let mut marks = String::new();
    marks.push_str(&axis_svg(rect, range, &theme));
    let mut mean_series = BTreeMap::<String, Vec<(usize, Option<f64>)>>::new();
    let mut upper_series = BTreeMap::<String, Vec<(usize, Option<f64>)>>::new();
    let mut lower_series = BTreeMap::<String, Vec<(usize, Option<f64>)>>::new();
    for row in 0..table.row_count() {
        let period = periods[row].clone().unwrap_or_default();
        mean_series
            .entry(period.clone())
            .or_default()
            .push((row, means[row]));
        let band = means[row]
            .zip(errors.get(row).copied().flatten())
            .map(|(mean, error)| {
                Ok::<_, RenderError>((
                    checked_visual(mean + error, "spread band")?,
                    checked_visual(mean - error, "spread band")?,
                ))
            })
            .transpose()?;
        upper_series
            .entry(period.clone())
            .or_default()
            .push((row, band.map(|value| value.0)));
        lower_series
            .entry(period)
            .or_default()
            .push((row, band.map(|value| value.1)));
    }
    for (period, points) in mean_series {
        let color = Theme::series_color_for_key(&period).hex();
        if let (Some(upper), Some(lower)) = (upper_series.get(&period), lower_series.get(&period)) {
            marks.push_str(&line_paths(
                "band-upper",
                &period,
                upper,
                table.row_count(),
                range,
                rect,
                &color,
            )?);
            marks.push_str(&line_paths(
                "band-lower",
                &period,
                lower,
                table.row_count(),
                range,
                rect,
                &color,
            )?);
        }
        marks.push_str(&line_paths(
            "line",
            &period,
            &points,
            table.row_count(),
            range,
            rect,
            &color,
        )?);
    }
    Ok((chart_svg(panel, width, theme, &marks), Vec::new()))
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
        r#"<rect data-role="plot-body" x="{}" y="{}" width="{}" height="{}" fill="none" stroke="{}"/><line data-role="zero-line" x1="{}" y1="{:.3}" x2="{}" y2="{:.3}" stroke="{}" stroke-width="1"/><text data-role="axis-unit" x="{}" y="{}" fill="{}" font-size="11">{}</text>"#,
        rect.x,
        rect.y,
        rect.width,
        rect.height,
        theme.grid.hex(),
        rect.x,
        zero,
        rect.x + rect.width,
        zero,
        theme.grid.hex(),
        rect.x,
        rect.y + rect.height + 28,
        theme.muted_text.hex(),
        escape_xml(&format_number(0.0, DisplayUnit::DecimalReturn, 0))
    )
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
            checked_visual(*value, "line path")?;
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

fn kde_density(values: &[Option<f64>]) -> RenderResult<Vec<(f64, f64)>> {
    let mut finite = values.iter().copied().flatten().collect::<Vec<_>>();
    finite.sort_by(f64::total_cmp);
    if finite.is_empty() {
        return Ok(Vec::new());
    }
    let unique_count = 1 + finite
        .windows(2)
        .filter(|pair| pair[0].total_cmp(&pair[1]).is_ne())
        .count();
    if finite.len() == 1 || unique_count == 1 {
        return Ok(vec![(finite[0], 1.0)]);
    }
    for value in &finite {
        checked_visual(*value, "kde input")?;
    }
    let n = finite.len() as f64;
    let mean = finite.iter().sum::<f64>() / n;
    let variance = finite
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / (n - 1.0);
    let std_dev = checked_visual(variance.sqrt(), "kde bandwidth")?;
    let raw_bandwidth = 1.06 * std_dev * n.powf(-0.2);
    let span = checked_visual(finite[finite.len() - 1] - finite[0], "kde span")?;
    let bandwidth = if raw_bandwidth > 0.0 {
        raw_bandwidth
    } else {
        span / 10.0
    };
    checked_visual(bandwidth, "kde bandwidth")?;
    if bandwidth <= 0.0 {
        return Ok(vec![(finite[0], 1.0)]);
    }
    let low = checked_visual(finite[0] - 3.0 * bandwidth, "kde bounds")?;
    let high = checked_visual(finite[finite.len() - 1] + 3.0 * bandwidth, "kde bounds")?;
    let mut density = Vec::with_capacity(KDE_POINTS);
    let normalizer = n * bandwidth * (2.0 * std::f64::consts::PI).sqrt();
    for index in 0..KDE_POINTS {
        let x = low + (high - low) * index as f64 / (KDE_POINTS - 1) as f64;
        let y = finite
            .iter()
            .map(|value| {
                let z = (x - *value) / bandwidth;
                (-0.5 * z * z).exp()
            })
            .sum::<f64>()
            / normalizer;
        density.push((checked_visual(x, "kde x")?, checked_visual(y, "kde y")?));
    }
    let max = density.iter().map(|(_, y)| *y).fold(0.0, f64::max);
    if max <= 0.0 || !max.is_finite() {
        return Err(RenderError::InvalidContract {
            reason: "kde density produced non-finite visual value".to_string(),
        });
    }
    for (_, y) in &mut density {
        *y = checked_visual(*y / max, "kde normalize")?;
    }
    Ok(density)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kde_density_is_stable_sorted_and_normalized() {
        let forward = kde_density(&[Some(0.0), Some(1.0), Some(2.0)]).unwrap();
        let reverse = kde_density(&[Some(2.0), Some(1.0), Some(0.0)]).unwrap();

        assert_eq!(forward.len(), 64);
        assert_eq!(forward, reverse);
        assert!(forward.windows(2).all(|pair| pair[0].0 < pair[1].0));
        assert!(forward.iter().all(|(_, y)| y.is_finite() && *y >= 0.0));
        let max_density = forward.iter().map(|(_, y)| *y).fold(0.0, f64::max);
        assert_eq!(max_density, 1.0);
    }

    #[test]
    fn kde_density_handles_degenerate_inputs_and_rejects_overflow() {
        assert!(kde_density(&[None, None]).unwrap().is_empty());
        assert_eq!(kde_density(&[Some(3.0)]).unwrap().len(), 1);
        assert_eq!(kde_density(&[Some(4.0), Some(4.0)]).unwrap().len(), 1);
        assert!(kde_density(&[Some(f64::MAX), Some(f64::MAX / 2.0)]).is_err());
    }
}
