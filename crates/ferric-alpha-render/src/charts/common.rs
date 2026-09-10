use plotters::coord::Shift;
use plotters::prelude::{DrawingArea, DrawingBackend, IntoFont, Text, TextStyle};
use plotters::style::text_anchor::{HPos, Pos, VPos};

use crate::{FONT_FAMILY, RenderError, RenderResult, ensure_font_registered};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextAllocation {
    pub id: String,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub raw_width: u32,
    pub measured_width: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

pub fn checked_padded_range(
    values: impl Iterator<Item = Option<f64>>,
    include_zero: bool,
) -> RenderResult<Option<(f64, f64)>> {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut found = false;
    for value in values.flatten() {
        checked_visual(value, "padded_range")?;
        min = min.min(value);
        max = max.max(value);
        found = true;
    }
    if !found {
        return Ok(None);
    }
    if include_zero {
        min = min.min(0.0);
        max = max.max(0.0);
    }
    let (low, high) = if min == max {
        let pad = if min == 0.0 {
            1.0
        } else {
            min.abs().max(1.0) * 0.1
        };
        (min - pad, max + pad)
    } else {
        let pad = (max - min) * 0.1;
        (min - pad, max + pad)
    };
    checked_visual(low, "padded_range")?;
    checked_visual(high, "padded_range")?;
    Ok(Some((round_for_stability(low), round_for_stability(high))))
}

pub fn date_tick_indices(length: usize, maximum: usize) -> Vec<usize> {
    if length == 0 || maximum == 0 {
        return Vec::new();
    }
    if length <= maximum {
        return (0..length).collect();
    }
    let last = length - 1;
    (0..maximum)
        .map(|index| ((index * last) + (maximum - 1) / 2) / (maximum - 1))
        .collect()
}

pub fn escape_xml(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

pub fn measure_and_fit(value: &str, width: u32, font_size: u32) -> String {
    if width == 0 {
        return String::new();
    }
    if measured_width(value, font_size) <= width {
        return value.to_string();
    }
    let ellipsis = "…";
    let ellipsis_width = measured_width(ellipsis, font_size);
    if ellipsis_width >= width {
        return ellipsis.to_string();
    }
    let mut out = String::new();
    for character in value.chars() {
        let mut candidate = out.clone();
        candidate.push(character);
        candidate.push_str(ellipsis);
        if measured_width(&candidate, font_size) > width {
            break;
        }
        out.push(character);
    }
    out.push_str(ellipsis);
    out
}

pub fn draw_text_in_rect<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    allocation: TextAllocation,
    value: &str,
    style: &TextStyle<'_>,
    manifest: &mut Vec<TextAllocation>,
) -> RenderResult<()>
where
    DB::ErrorType: std::error::Error + Send + Sync + 'static,
{
    let font_size = style.font.get_size().round() as u32;
    let fitted = measure_and_fit(value, allocation.width.saturating_sub(4), font_size);
    manifest.push(allocation.clone());
    let style = style.clone().pos(Pos::new(HPos::Left, VPos::Center));
    let x =
        i32::try_from(allocation.x.saturating_add(2)).map_err(|_| RenderError::LayoutOverflow)?;
    let y = i32::try_from(allocation.y + allocation.height / 2)
        .map_err(|_| RenderError::LayoutOverflow)?;
    area.draw(&Text::new(fitted, (x, y), style))
        .map_err(|error| RenderError::Backend {
            format: "SVG",
            message: error.to_string(),
        })?;
    Ok(())
}

pub fn checked_visual(value: f64, operation: &'static str) -> RenderResult<f64> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(RenderError::InvalidContract {
            reason: format!("{operation} produced non-finite visual value"),
        })
    }
}

pub(crate) fn text_allocation(
    id: String,
    rect: Rect,
    value: &str,
    font_size: u32,
) -> TextAllocation {
    let raw_width = measured_width(value, font_size);
    let measured_width = raw_width.saturating_mul(5) / 4 + 4;
    TextAllocation {
        id,
        x: rect.x,
        y: rect.y,
        width: rect.width,
        height: rect.height,
        raw_width,
        measured_width,
    }
}

pub(crate) fn text_style(color: crate::Color, size: u32) -> RenderResult<TextStyle<'static>> {
    ensure_font_registered()?;
    Ok((FONT_FAMILY, size).into_font().color(&color.rgb()))
}

fn measured_width(value: &str, font_size: u32) -> u32 {
    if ensure_font_registered().is_ok()
        && let Ok((width, _)) = (FONT_FAMILY, font_size).into_font().box_size(value)
    {
        return width;
    }
    value
        .chars()
        .map(|ch| if ch.is_ascii() { 7 } else { 14 })
        .sum()
}

fn round_for_stability(value: f64) -> f64 {
    (value * 1_000_000_000_000.0).round() / 1_000_000_000_000.0
}
