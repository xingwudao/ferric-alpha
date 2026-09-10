use crate::RenderTheme;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Color {
    pub red: u8,
    pub green: u8,
    pub blue: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Theme {
    pub background: Color,
    pub surface: Color,
    pub alternate_row: Color,
    pub grid: Color,
    pub text: Color,
    pub muted_text: Color,
    pub accent: Color,
}

const SERIES: [Color; 7] = [
    Color::new(0x25, 0x63, 0xeb),
    Color::new(0xdc, 0x26, 0x26),
    Color::new(0x16, 0xa3, 0x4a),
    Color::new(0x93, 0x33, 0xea),
    Color::new(0xea, 0x58, 0x0c),
    Color::new(0x08, 0x94, 0x94),
    Color::new(0xbe, 0x12, 0x3c),
];

impl Color {
    pub const fn new(red: u8, green: u8, blue: u8) -> Self {
        Self { red, green, blue }
    }

    pub fn hex(self) -> String {
        format!("#{:02x}{:02x}{:02x}", self.red, self.green, self.blue)
    }

    pub(crate) fn rgb(self) -> plotters::style::RGBColor {
        plotters::style::RGBColor(self.red, self.green, self.blue)
    }
}

impl Theme {
    pub fn series_color_for_key(key: &str) -> Color {
        let index = stable_index(key, SERIES.len());
        SERIES[index]
    }
}

pub fn theme_for(theme: RenderTheme) -> Theme {
    match theme {
        RenderTheme::Light => Theme {
            background: Color::new(0xff, 0xff, 0xff),
            surface: Color::new(0xff, 0xff, 0xff),
            alternate_row: Color::new(0xf8, 0xfa, 0xfc),
            grid: Color::new(0xe5, 0xe7, 0xeb),
            text: Color::new(0x1f, 0x29, 0x37),
            muted_text: Color::new(0x6b, 0x72, 0x80),
            accent: Color::new(0x25, 0x63, 0xeb),
        },
        RenderTheme::Dark => Theme {
            background: Color::new(0x11, 0x18, 0x27),
            surface: Color::new(0x1f, 0x29, 0x37),
            alternate_row: Color::new(0x17, 0x24, 0x36),
            grid: Color::new(0x37, 0x41, 0x51),
            text: Color::new(0xf9, 0xfa, 0xfb),
            muted_text: Color::new(0xd1, 0xd5, 0xdb),
            accent: Color::new(0x60, 0xa5, 0xfa),
        },
    }
}

pub fn series_color(index: usize) -> Color {
    SERIES[index % SERIES.len()]
}

pub fn contrast_ratio(left: Color, right: Color) -> f64 {
    let left = relative_luminance(left);
    let right = relative_luminance(right);
    let lighter = left.max(right);
    let darker = left.min(right);
    (lighter + 0.05) / (darker + 0.05)
}

fn relative_luminance(color: Color) -> f64 {
    fn channel(value: u8) -> f64 {
        let value = f64::from(value) / 255.0;
        if value <= 0.03928 {
            value / 12.92
        } else {
            ((value + 0.055) / 1.055).powf(2.4)
        }
    }
    0.2126 * channel(color.red) + 0.7152 * channel(color.green) + 0.0722 * channel(color.blue)
}

fn stable_index(value: &str, modulo: usize) -> usize {
    let hash = value.bytes().fold(0xcbf29ce484222325_u64, |hash, byte| {
        (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
    });
    hash as usize % modulo
}
