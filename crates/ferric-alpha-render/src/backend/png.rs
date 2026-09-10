use std::io::{self, Write};
use std::path::Path;

use ferric_alpha::{ReportColumnData, ReportTable, TearSheetData};
use plotters::coord::Shift;
use plotters::prelude::{
    BitMapBackend, Circle, DrawingArea, DrawingBackend, IntoDrawingArea, PathElement, Rectangle,
    Text,
};
use plotters::style::{IntoFont, ShapeStyle};

use crate::charts::table::draw_table_panel;
use crate::{
    FONT_FAMILY, MAX_PNG_BYTES, PanelKind, PanelPlan, RenderError, RenderOptions, RenderResult,
    Theme, checked_padded_range, checked_visual, ensure_font_registered, plan_report, series_color,
    theme_for,
};

pub struct LimitedVecWriter {
    limit: usize,
    bytes: Vec<u8>,
    failure: Option<LimitedVecFailure>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LimitedVecFailure {
    OutputTooLarge,
    Allocation { bytes: usize },
}

impl LimitedVecWriter {
    pub fn new(limit: usize) -> Self {
        Self {
            limit,
            bytes: Vec::new(),
            failure: None,
        }
    }

    pub fn finish(self) -> RenderResult<Vec<u8>> {
        match self.failure {
            Some(LimitedVecFailure::OutputTooLarge) => Err(RenderError::OutputTooLarge {
                format: "PNG",
                limit: self.limit,
            }),
            Some(LimitedVecFailure::Allocation { bytes }) => Err(RenderError::Allocation { bytes }),
            None => Ok(self.bytes),
        }
    }
}

impl Write for LimitedVecWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if self.failure.is_some() {
            return Err(io::Error::other("limited writer is already failed"));
        }
        let Some(next_len) = self.bytes.len().checked_add(buf.len()) else {
            self.failure = Some(LimitedVecFailure::OutputTooLarge);
            return Err(io::Error::other("limited writer length overflow"));
        };
        if next_len > self.limit {
            self.failure = Some(LimitedVecFailure::OutputTooLarge);
            return Err(io::Error::other("limited writer output too large"));
        }
        if self.bytes.try_reserve_exact(buf.len()).is_err() {
            self.failure = Some(LimitedVecFailure::Allocation { bytes: buf.len() });
            return Err(io::Error::other("limited writer allocation failed"));
        }
        self.bytes.extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub fn render_png(report: &TearSheetData, options: &RenderOptions) -> RenderResult<Vec<u8>> {
    let plan = plan_report(report, options)?;
    let physical = options.validate(plan.height)?;
    let pixels = usize::try_from(
        u64::from(physical.width)
            .checked_mul(u64::from(physical.height))
            .and_then(|pixels| pixels.checked_mul(3))
            .ok_or(RenderError::LayoutOverflow)?,
    )
    .map_err(|_| RenderError::LayoutOverflow)?;
    let mut rgb = Vec::new();
    rgb.try_reserve_exact(pixels)
        .map_err(|_| RenderError::Allocation { bytes: pixels })?;
    rgb.resize(pixels, 255);

    {
        ensure_font_registered()?;
        let theme = theme_for(options.theme);
        let backend = BitMapBackend::with_buffer(&mut rgb, (physical.width, physical.height));
        let area = backend.into_drawing_area();
        area.fill(&theme.background.rgb())
            .map_err(|error| RenderError::Backend {
                format: "PNG",
                message: error.to_string(),
            })?;
        let scale = options.scale;
        let text_style = (FONT_FAMILY, (18.0 * scale).round() as u32)
            .into_font()
            .color(&theme.text.rgb());
        area.draw(&Text::new(
            plan.title.clone(),
            ((24.0 * scale) as i32, (36.0 * scale) as i32),
            text_style,
        ))
        .map_err(|error| RenderError::Backend {
            format: "PNG",
            message: error.to_string(),
        })?;
        for panel in &plan.panels {
            let x = (panel_x(plan.width, panel) as f64 * scale).round() as i32;
            let y = (panel_y(&plan.panels, panel) as f64 * scale).round() as i32;
            let w = (panel_width(plan.width, panel) as f64 * scale).round() as i32;
            let h = (panel.height as f64 * scale).round() as i32;
            area.draw(&Rectangle::new(
                [(x, y), (x + w, y + h)],
                ShapeStyle::from(&theme.surface.rgb()).filled(),
            ))
            .map_err(|error| RenderError::Backend {
                format: "PNG",
                message: error.to_string(),
            })?;
            area.draw(&Rectangle::new(
                [(x, y), (x + w, y + h)],
                ShapeStyle::from(&theme.grid.rgb()).stroke_width(1),
            ))
            .map_err(|error| RenderError::Backend {
                format: "PNG",
                message: error.to_string(),
            })?;
            let panel_area = area.clone().shrink((x, y), (w.max(1), h.max(1)));
            draw_png_panel(&panel_area, report, panel, &theme)?;
        }
        area.present().map_err(|error| RenderError::Backend {
            format: "PNG",
            message: error.to_string(),
        })?;
    }

    let mut output = LimitedVecWriter::new(MAX_PNG_BYTES);
    let mut encoder = png::Encoder::new(&mut output, physical.width, physical.height);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder
        .write_header()
        .map_err(|error| RenderError::PngEncoding {
            message: error.to_string(),
        })?;
    writer
        .write_image_data(&rgb)
        .map_err(|error| RenderError::PngEncoding {
            message: error.to_string(),
        })?;
    writer.finish().map_err(|error| RenderError::PngEncoding {
        message: error.to_string(),
    })?;
    output.finish()
}

pub fn write_png(
    report: &TearSheetData,
    options: &RenderOptions,
    path: impl AsRef<Path>,
) -> RenderResult<()> {
    let path = path.as_ref();
    let bytes = render_png(report, options)?;
    std::fs::write(path, bytes).map_err(|source| RenderError::Io {
        path: path.to_path_buf(),
        source,
    })
}

fn panel_x(plan_width: u32, panel: &crate::PanelPlan) -> u32 {
    let margin = 24;
    let content_width = plan_width - 48;
    if panel.column == 0 {
        margin
    } else {
        margin + content_width / 2 + 8
    }
}

fn panel_width(plan_width: u32, panel: &crate::PanelPlan) -> u32 {
    let content = plan_width - 48;
    if panel.column_span == 12 {
        content
    } else {
        (content - 16) / 2
    }
}

fn panel_y(panels: &[crate::PanelPlan], panel: &crate::PanelPlan) -> u32 {
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

fn draw_png_panel<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    report: &TearSheetData,
    panel: &PanelPlan,
    theme: &Theme,
) -> RenderResult<()>
where
    DB::ErrorType: std::error::Error + Send + Sync + 'static,
{
    match panel.kind {
        PanelKind::Table => {
            let tables = panel
                .table_ids
                .iter()
                .map(|id| {
                    report
                        .table(id)
                        .ok_or_else(|| RenderError::MissingTable { id: id.to_string() })
                })
                .collect::<RenderResult<Vec<_>>>()?;
            let mut manifest = Vec::new();
            draw_table_panel(area, &panel.id, &panel.title, &tables, theme, &mut manifest)
        }
        _ => draw_chart_panel(area, report, panel, theme),
    }
}

fn draw_chart_panel<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    report: &TearSheetData,
    panel: &PanelPlan,
    theme: &Theme,
) -> RenderResult<()>
where
    DB::ErrorType: std::error::Error + Send + Sync + 'static,
{
    area.fill(&theme.surface.rgb())
        .map_err(|error| backend_error(error.to_string()))?;
    let (width, height) = area.dim_in_pixel();
    area.draw(&Rectangle::new(
        [(0, 0), (width as i32, height as i32)],
        ShapeStyle::from(&theme.grid.rgb()).stroke_width(1),
    ))
    .map_err(|error| backend_error(error.to_string()))?;
    area.draw(&Text::new(
        panel.title.clone(),
        (16, 26),
        (FONT_FAMILY, 16).into_font().color(&theme.text.rgb()),
    ))
    .map_err(|error| backend_error(error.to_string()))?;

    let Some(table) = panel.table_ids.first().and_then(|id| report.table(id)) else {
        return Ok(());
    };
    let values = preferred_numeric_values(table, &panel.kind);
    if values.is_empty() {
        return Ok(());
    }
    let rect = PlotRect {
        x: 64,
        y: 48,
        width: width.saturating_sub(88).max(1),
        height: height.saturating_sub(94).max(1),
    };
    let range = checked_padded_range(values.iter().copied(), true)?.unwrap_or((-1.0, 1.0));
    draw_axis(area, rect, range, theme)?;
    match panel.kind {
        PanelKind::QuantileReturnsBar { .. }
        | PanelKind::QuantileReturnsDistribution { .. }
        | PanelKind::IcHistogram { .. }
        | PanelKind::IcByGroup
        | PanelKind::EventDistribution => draw_bars(area, &values, rect, range),
        PanelKind::IcQq => draw_points(area, table, rect, range, theme),
        _ => draw_line(area, &values, rect, range, theme),
    }
}

#[derive(Debug, Clone, Copy)]
struct PlotRect {
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

fn preferred_numeric_values(table: &ReportTable, kind: &PanelKind) -> Vec<Option<f64>> {
    let preferred = match kind {
        PanelKind::QuantileReturnsBar { .. } => &["mean_return"][..],
        PanelKind::QuantileReturnsDistribution { .. } => &["mean_return"][..],
        PanelKind::CumulativeReturns { .. } => &["cumulative_return"][..],
        PanelKind::MeanSpread => &["mean_return_difference"][..],
        PanelKind::IcTimeSeries => &["ic", "rolling_mean_ic"][..],
        PanelKind::IcHistogram { .. } => &["ic"][..],
        PanelKind::IcQq => &["observed"][..],
        PanelKind::IcMonthlyHeatmap => &["mean_ic"][..],
        PanelKind::IcByGroup => &["mean_ic"][..],
        PanelKind::QuantileTurnover { .. } => &["turnover"][..],
        PanelKind::RankAutocorrelation { .. } => &["autocorrelation"][..],
        PanelKind::EventDistribution => &["event_count"][..],
        PanelKind::EventAverageCumulative { .. } => &["mean_cumulative_return"][..],
        PanelKind::Table => &[][..],
    };
    for name in preferred {
        if let Some(values) = float_column(table, name) {
            return values;
        }
    }
    table
        .columns()
        .iter()
        .find_map(|column| match column.data() {
            ReportColumnData::Float64(values) => Some(values.clone()),
            ReportColumnData::Int32(values) => {
                Some(values.iter().map(|v| v.map(f64::from)).collect())
            }
            ReportColumnData::UInt32(values) => {
                Some(values.iter().map(|v| v.map(f64::from)).collect())
            }
            ReportColumnData::UInt64(values) => {
                Some(values.iter().map(|v| v.map(|v| v as f64)).collect())
            }
            _ => None,
        })
        .unwrap_or_default()
}

fn float_column(table: &ReportTable, name: &str) -> Option<Vec<Option<f64>>> {
    table
        .columns()
        .iter()
        .find(|column| column.name() == name)
        .and_then(|column| match column.data() {
            ReportColumnData::Float64(values) => Some(values.clone()),
            ReportColumnData::Int32(values) => {
                Some(values.iter().map(|v| v.map(f64::from)).collect())
            }
            ReportColumnData::UInt32(values) => {
                Some(values.iter().map(|v| v.map(f64::from)).collect())
            }
            ReportColumnData::UInt64(values) => {
                Some(values.iter().map(|v| v.map(|v| v as f64)).collect())
            }
            _ => None,
        })
}

fn draw_axis<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    rect: PlotRect,
    range: (f64, f64),
    theme: &Theme,
) -> RenderResult<()>
where
    DB::ErrorType: std::error::Error + Send + Sync + 'static,
{
    area.draw(&Rectangle::new(
        [
            (rect.x as i32, rect.y as i32),
            ((rect.x + rect.width) as i32, (rect.y + rect.height) as i32),
        ],
        ShapeStyle::from(&theme.grid.rgb()).stroke_width(1),
    ))
    .map_err(|error| backend_error(error.to_string()))?;
    let zero = scale_y(0.0, range, rect);
    area.draw(&PathElement::new(
        vec![(rect.x as i32, zero), ((rect.x + rect.width) as i32, zero)],
        ShapeStyle::from(&theme.grid.rgb()).stroke_width(1),
    ))
    .map_err(|error| backend_error(error.to_string()))
}

fn draw_bars<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    values: &[Option<f64>],
    rect: PlotRect,
    range: (f64, f64),
) -> RenderResult<()>
where
    DB::ErrorType: std::error::Error + Send + Sync + 'static,
{
    let lanes = values.len().max(1) as f64;
    let lane_width = rect.width as f64 / lanes;
    let zero = scale_y(0.0, range, rect);
    for (index, value) in values.iter().enumerate() {
        let Some(value) = *value else { continue };
        checked_visual(value, "png bar")?;
        let x = rect.x as f64 + index as f64 * lane_width + lane_width * 0.15;
        let y = scale_y(value, range, rect);
        area.draw(&Rectangle::new(
            [
                (x.round() as i32, zero.min(y)),
                ((x + lane_width * 0.7).round() as i32, zero.max(y)),
            ],
            ShapeStyle::from(&series_color(index).rgb()).filled(),
        ))
        .map_err(|error| backend_error(error.to_string()))?;
    }
    Ok(())
}

fn draw_line<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    values: &[Option<f64>],
    rect: PlotRect,
    range: (f64, f64),
    theme: &Theme,
) -> RenderResult<()>
where
    DB::ErrorType: std::error::Error + Send + Sync + 'static,
{
    let mut points = Vec::new();
    for (index, value) in values.iter().enumerate() {
        if let Some(value) = *value {
            checked_visual(value, "png line")?;
            points.push((
                scale_x(index, values.len(), rect),
                scale_y(value, range, rect),
            ));
        }
    }
    if points.len() >= 2 {
        area.draw(&PathElement::new(
            points,
            ShapeStyle::from(&theme.accent.rgb()).stroke_width(2),
        ))
        .map_err(|error| backend_error(error.to_string()))?;
    }
    Ok(())
}

fn draw_points<DB: DrawingBackend>(
    area: &DrawingArea<DB, Shift>,
    table: &ReportTable,
    rect: PlotRect,
    range: (f64, f64),
    theme: &Theme,
) -> RenderResult<()>
where
    DB::ErrorType: std::error::Error + Send + Sync + 'static,
{
    let xs = float_column(table, "theoretical").unwrap_or_default();
    let ys = float_column(table, "observed").unwrap_or_default();
    for (x_value, y_value) in xs.iter().zip(ys.iter()) {
        if let (Some(x_value), Some(y_value)) = (*x_value, *y_value) {
            area.draw(&Circle::new(
                (
                    scale_value_x(x_value, range, rect),
                    scale_y(y_value, range, rect),
                ),
                3,
                ShapeStyle::from(&theme.accent.rgb()).filled(),
            ))
            .map_err(|error| backend_error(error.to_string()))?;
        }
    }
    Ok(())
}

fn scale_x(index: usize, total: usize, rect: PlotRect) -> i32 {
    if total <= 1 {
        return (rect.x + rect.width / 2) as i32;
    }
    (rect.x as f64 + index as f64 / (total - 1) as f64 * rect.width as f64).round() as i32
}

fn scale_value_x(value: f64, range: (f64, f64), rect: PlotRect) -> i32 {
    let ratio = if range.0 == range.1 {
        0.5
    } else {
        (value - range.0) / (range.1 - range.0)
    };
    (rect.x as f64 + ratio.clamp(0.0, 1.0) * rect.width as f64).round() as i32
}

fn scale_y(value: f64, range: (f64, f64), rect: PlotRect) -> i32 {
    let ratio = if range.0 == range.1 {
        0.5
    } else {
        (value - range.0) / (range.1 - range.0)
    };
    (rect.y as f64 + (1.0 - ratio.clamp(0.0, 1.0)) * rect.height as f64).round() as i32
}

fn backend_error(message: String) -> RenderError {
    RenderError::Backend {
        format: "PNG",
        message,
    }
}
