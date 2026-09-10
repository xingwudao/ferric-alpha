use serde::{Deserialize, Serialize};

use crate::{RenderError, RenderResult};

pub const MIN_WIDTH: u32 = 720;
pub const MAX_WIDTH: u32 = 4096;
pub const MAX_PHYSICAL_SIDE: u32 = 8_192;
pub const MAX_LOGICAL_HEIGHT: u32 = 32_768;
pub const MAX_PANELS: usize = 256;
pub const MAX_PNG_PIXELS: u64 = 16_000_000;
pub const MAX_PNG_BYTES: usize = 64 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RenderTheme {
    Light,
    Dark,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RenderOptions {
    pub width: u32,
    pub scale: f64,
    pub theme: RenderTheme,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PhysicalSize {
    pub width: u32,
    pub height: u32,
}

impl Default for RenderOptions {
    fn default() -> Self {
        Self {
            width: 1200,
            scale: 1.0,
            theme: RenderTheme::Light,
        }
    }
}

impl RenderOptions {
    pub fn validate(&self, logical_height: u32) -> RenderResult<PhysicalSize> {
        if !(MIN_WIDTH..=MAX_WIDTH).contains(&self.width) {
            return Err(RenderError::InvalidOption {
                option: "width",
                reason: "must be in 720..=4096",
            });
        }
        if !self.scale.is_finite() || !(0.5..=4.0).contains(&self.scale) {
            return Err(RenderError::InvalidOption {
                option: "scale",
                reason: "must be finite and in 0.5..=4.0",
            });
        }
        if logical_height > MAX_LOGICAL_HEIGHT {
            return Err(RenderError::LayoutOverflow);
        }

        let physical_width = checked_physical_dimension(self.width, self.scale)?;
        let physical_height = checked_physical_dimension(logical_height, self.scale)?;
        if physical_width > MAX_PHYSICAL_SIDE || physical_height > MAX_PHYSICAL_SIDE {
            return Err(RenderError::LayoutOverflow);
        }

        let pixels = u64::from(physical_width)
            .checked_mul(u64::from(physical_height))
            .ok_or(RenderError::LayoutOverflow)?;
        if pixels > MAX_PNG_PIXELS {
            return Err(RenderError::LayoutOverflow);
        }

        Ok(PhysicalSize {
            width: physical_width,
            height: physical_height,
        })
    }
}

fn checked_physical_dimension(logical: u32, scale: f64) -> RenderResult<u32> {
    let scaled = f64::from(logical) * scale;
    if !scaled.is_finite() || scaled < 0.0 || scaled > f64::from(u32::MAX) {
        return Err(RenderError::LayoutOverflow);
    }
    Ok(scaled.round() as u32)
}
