use std::sync::OnceLock;

use plotters::style::{FontStyle, register_font};

use crate::{RenderError, RenderResult};

pub const FONT_FAMILY: &str = "Ferric Noto Sans SC";

static FONT_REGISTRATION: OnceLock<()> = OnceLock::new();

pub fn ensure_font_registered() -> RenderResult<()> {
    if FONT_REGISTRATION.get().is_some() {
        return Ok(());
    }

    register_font(FONT_FAMILY, FontStyle::Normal, font_bytes()).map_err(|_| {
        RenderError::Backend {
            format: "font",
            message: "failed to register bundled Noto Sans SC".to_string(),
        }
    })?;
    let _ = FONT_REGISTRATION.set(());
    Ok(())
}

#[doc(hidden)]
pub fn font_bytes() -> &'static [u8] {
    include_bytes!("../assets/fonts/NotoSansSC-wght.ttf")
}

#[doc(hidden)]
pub fn font_license() -> &'static str {
    include_str!("../assets/fonts/OFL.txt")
}

pub fn font_sha256() -> &'static str {
    "a3041811a78c361b1de50f953c805e0244951c21c5bd412f7232ef0d899af0da"
}
