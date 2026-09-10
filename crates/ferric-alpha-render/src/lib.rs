#![forbid(unsafe_code)]

mod data;
mod error;
mod font;
mod format;
mod options;
mod plan;
mod theme;

mod backend;
mod charts;

pub const RENDER_PLAN_VERSION: &str = "ferric-alpha.render-plan/v1";

pub use data::{
    DatetimeColumn, bools, datetimes, float64, int32, int64, require_table, strings, uint32, uint64,
};
pub use error::{RenderError, RenderResult};
pub use font::{FONT_FAMILY, ensure_font_registered, font_bytes, font_license, font_sha256};
pub use format::{format_axis_value, format_cell, format_datetime, format_number, format_optional};
pub use options::{
    MAX_LOGICAL_HEIGHT, MAX_PANELS, MAX_PHYSICAL_SIDE, MAX_PNG_BYTES, MAX_PNG_PIXELS, MAX_WIDTH,
    MIN_WIDTH, PhysicalSize, RenderOptions, RenderTheme,
};
pub use plan::{PanelKind, PanelPlan, RenderPlan, plan_report};
pub use theme::{Color, Theme, contrast_ratio, series_color, theme_for};

pub use backend::{
    LimitedVecWriter, RenderedPanel, render_html, render_html_fragment, render_panel_svg,
    render_png, render_svg, render_table_panel_svg, render_table_svg, write_html, write_png,
    write_svg,
};
pub use charts::common::{
    TextAllocation, checked_padded_range, checked_visual, date_tick_indices, escape_xml,
    measure_and_fit,
};
