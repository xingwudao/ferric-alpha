use std::sync::OnceLock;

use plotters::style::{FontStyle, register_font};

use crate::{RenderError, RenderResult};

pub const FONT_FAMILY: &str = "Roboto";

static FONT_REGISTRATION: OnceLock<()> = OnceLock::new();

pub fn ensure_font_registered() -> RenderResult<()> {
    if FONT_REGISTRATION.get().is_some() {
        return Ok(());
    }

    register_font(FONT_FAMILY, FontStyle::Normal, font_bytes()).map_err(|_| {
        RenderError::Backend {
            format: "font",
            message: "failed to register bundled Roboto font".to_string(),
        }
    })?;
    let _ = FONT_REGISTRATION.set(());
    Ok(())
}

#[doc(hidden)]
pub fn font_bytes() -> &'static [u8] {
    include_bytes!("../assets/fonts/RobotoStatic-Regular.ttf")
}

#[doc(hidden)]
pub fn font_license() -> &'static str {
    include_str!("../assets/fonts/Roboto-NOTICE.txt")
}

pub fn font_sha256() -> &'static str {
    "06cba01eb71ea5cbd3a7df498910624db68953beead4be18fd91f8ec7dc72351"
}
