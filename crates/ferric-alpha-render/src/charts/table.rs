use ferric_alpha::{ColumnRole, DisplayUnit, ReportColumn, ReportColumnData, ReportTable};
use plotters::coord::Shift;
use plotters::prelude::{DrawingArea, DrawingBackend, Rectangle};
use plotters::style::ShapeStyle;

use crate::charts::common::{Rect, TextAllocation, draw_text_in_rect, text_allocation, text_style};
use crate::format::{format_datetime, format_number};
use crate::{RenderError, RenderResult, Theme};

const TITLE_HEIGHT: u32 = 36;
const TABLE_GAP: u32 = 10;
const HEADER_HEIGHT: u32 = 30;
const ROW_HEIGHT: u32 = 30;
const CELL_PADDING: u32 = 8;

struct TableDrawContext<'a> {
    panel_id: &'a str,
    table_index: usize,
    y: u32,
    width: u32,
    theme: &'a Theme,
}

pub(crate) fn draw_table_panel<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    panel_id: &str,
    title: &str,
    tables: &[&ReportTable],
    theme: &Theme,
    manifest: &mut Vec<TextAllocation>,
) -> RenderResult<()>
where
    DB::ErrorType: std::error::Error + Send + Sync + 'static,
{
    area.fill(&theme.background.rgb())
        .map_err(|error| backend_error(error.to_string()))?;
    let (width, height) = area.dim_in_pixel();
    draw_rect(
        area,
        Rect {
            x: 0,
            y: 0,
            width,
            height,
        },
        theme.grid,
        false,
    )?;
    draw_text_in_rect(
        area,
        text_allocation(
            format!("{panel_id}.title"),
            Rect {
                x: 12,
                y: 0,
                width: width.saturating_sub(24),
                height: TITLE_HEIGHT,
            },
            title,
            17,
        ),
        title,
        &text_style(theme.text, 17)?,
        manifest,
    )?;

    let mut y = TITLE_HEIGHT;
    for (table_index, table) in tables.iter().enumerate() {
        if table_index > 0 {
            y = y.saturating_add(TABLE_GAP);
        }
        if tables.len() > 1 {
            draw_text_in_rect(
                area,
                text_allocation(
                    format!("{panel_id}.table-{table_index}.title"),
                    Rect {
                        x: 12,
                        y,
                        width: width.saturating_sub(24),
                        height: HEADER_HEIGHT,
                    },
                    table.title(),
                    12,
                ),
                table.title(),
                &text_style(theme.muted_text, 12)?,
                manifest,
            )?;
            y = y.saturating_add(HEADER_HEIGHT);
        }
        draw_one_table(
            area,
            table,
            TableDrawContext {
                panel_id,
                table_index,
                y,
                width,
                theme,
            },
            manifest,
        )?;
        y = y
            .saturating_add(HEADER_HEIGHT)
            .saturating_add(ROW_HEIGHT * table.row_count().max(1) as u32);
    }
    Ok(())
}

fn draw_one_table<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    table: &ReportTable,
    context: TableDrawContext<'_>,
    manifest: &mut Vec<TextAllocation>,
) -> RenderResult<()>
where
    DB::ErrorType: std::error::Error + Send + Sync + 'static,
{
    let columns = table.columns();
    if columns.is_empty() {
        return Err(RenderError::InvalidContract {
            reason: "table panel has no columns".to_string(),
        });
    }
    let column_width = context.width.saturating_sub(24) / columns.len() as u32;
    let start_x = 12;
    draw_rect(
        area,
        Rect {
            x: start_x,
            y: context.y,
            width: column_width * columns.len() as u32,
            height: HEADER_HEIGHT,
        },
        context.theme.grid,
        true,
    )?;
    for (column_index, column) in columns.iter().enumerate() {
        let rect = cell_rect(start_x, context.y, column_width, column_index);
        draw_text_in_rect(
            area,
            text_allocation(
                format!(
                    "{}.table-{}.header-{column_index}",
                    context.panel_id, context.table_index
                ),
                padded(rect),
                column.label(),
                12,
            ),
            column.label(),
            &text_style(context.theme.text, 12)?,
            manifest,
        )?;
    }

    if table.row_count() == 0 {
        let rect = Rect {
            x: start_x,
            y: context.y + HEADER_HEIGHT,
            width: column_width * columns.len() as u32,
            height: ROW_HEIGHT,
        };
        draw_rect(area, rect, context.theme.alternate_row, true)?;
        draw_text_in_rect(
            area,
            text_allocation(
                format!("{}.table-{}.empty", context.panel_id, context.table_index),
                padded(rect),
                "No rows",
                11,
            ),
            "No rows",
            &text_style(context.theme.muted_text, 11)?,
            manifest,
        )?;
        return Ok(());
    }

    for row in 0..table.row_count() {
        let row_y = context.y + HEADER_HEIGHT + row as u32 * ROW_HEIGHT;
        if row % 2 == 1 {
            draw_rect(
                area,
                Rect {
                    x: start_x,
                    y: row_y,
                    width: column_width * columns.len() as u32,
                    height: ROW_HEIGHT,
                },
                context.theme.alternate_row,
                true,
            )?;
        }
        for (column_index, column) in columns.iter().enumerate() {
            let value = format_column_cell(column, row)?;
            let rect = padded(cell_rect(start_x, row_y, column_width, column_index));
            let rect = if matches!(column.role(), ColumnRole::Measure | ColumnRole::Statistic) {
                right_aligned(rect, &value, 11)
            } else {
                rect
            };
            draw_text_in_rect(
                area,
                text_allocation(
                    format!(
                        "{}.table-{}.row-{row}.column-{column_index}",
                        context.panel_id, context.table_index
                    ),
                    rect,
                    &value,
                    11,
                ),
                &value,
                &text_style(context.theme.text, 11)?,
                manifest,
            )?;
        }
    }
    Ok(())
}

fn format_column_cell(column: &ReportColumn, row: usize) -> RenderResult<String> {
    let display = column.display();
    let value = match column.data() {
        ReportColumnData::Boolean(values) => values
            .get(row)
            .copied()
            .flatten()
            .map_or_else(|| "—".to_string(), |value| value.to_string()),
        ReportColumnData::Int32(values) => values
            .get(row)
            .copied()
            .flatten()
            .map_or_else(|| "—".to_string(), |value| value.to_string()),
        ReportColumnData::Int64(values) => values
            .get(row)
            .copied()
            .flatten()
            .map_or_else(|| "—".to_string(), |value| value.to_string()),
        ReportColumnData::UInt32(values) => values
            .get(row)
            .copied()
            .flatten()
            .map_or_else(|| "—".to_string(), |value| value.to_string()),
        ReportColumnData::UInt64(values) => values
            .get(row)
            .copied()
            .flatten()
            .map_or_else(|| "—".to_string(), |value| value.to_string()),
        ReportColumnData::Float64(values) => values.get(row).copied().flatten().map_or_else(
            || "—".to_string(),
            |value| match display {
                Some(display) => format_number(value, display.unit(), display.decimals()),
                None => format_number(value, DisplayUnit::Raw, 2),
            },
        ),
        ReportColumnData::String(values) => values
            .get(row)
            .and_then(Option::as_ref)
            .map_or_else(|| "—".to_string(), ToString::to_string),
        ReportColumnData::Datetime {
            values,
            time_unit,
            timezone,
        } => values.get(row).copied().flatten().map_or_else(
            || Ok("—".to_string()),
            |value| format_datetime(value, *time_unit, timezone.as_deref()),
        )?,
    };
    Ok(value)
}

fn cell_rect(start_x: u32, y: u32, column_width: u32, column_index: usize) -> Rect {
    Rect {
        x: start_x + column_width * column_index as u32,
        y,
        width: column_width,
        height: ROW_HEIGHT,
    }
}

fn padded(rect: Rect) -> Rect {
    Rect {
        x: rect.x + CELL_PADDING,
        y: rect.y,
        width: rect.width.saturating_sub(CELL_PADDING * 2),
        height: rect.height,
    }
}

fn right_aligned(rect: Rect, value: &str, font_size: u32) -> Rect {
    let estimated = value.chars().count() as u32 * font_size / 2 + 8;
    if estimated >= rect.width {
        rect
    } else {
        Rect {
            x: rect.x + rect.width - estimated,
            width: estimated,
            ..rect
        }
    }
}

fn draw_rect<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    rect: Rect,
    color: crate::Color,
    filled: bool,
) -> RenderResult<()>
where
    DB::ErrorType: std::error::Error + Send + Sync + 'static,
{
    let left = i32::try_from(rect.x).map_err(|_| RenderError::LayoutOverflow)?;
    let top = i32::try_from(rect.y).map_err(|_| RenderError::LayoutOverflow)?;
    let right = i32::try_from(rect.x + rect.width).map_err(|_| RenderError::LayoutOverflow)?;
    let bottom = i32::try_from(rect.y + rect.height).map_err(|_| RenderError::LayoutOverflow)?;
    let style = if filled {
        ShapeStyle::from(&color.rgb()).filled()
    } else {
        ShapeStyle::from(&color.rgb()).stroke_width(1)
    };
    area.draw(&Rectangle::new([(left, top), (right, bottom)], style))
        .map_err(|error| backend_error(error.to_string()))?;
    Ok(())
}

fn backend_error(message: String) -> RenderError {
    RenderError::Backend {
        format: "SVG",
        message,
    }
}
