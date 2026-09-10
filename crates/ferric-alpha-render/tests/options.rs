use ferric_alpha_render::{
    RenderError, RenderOptions, RenderTheme, ensure_font_registered, font_bytes, font_license,
    font_sha256,
};

#[test]
fn default_options_are_stable() {
    assert_eq!(
        RenderOptions::default(),
        RenderOptions {
            width: 1200,
            scale: 1.0,
            theme: RenderTheme::Light,
        }
    );
}

#[test]
fn options_reject_width_scale_and_pixel_overflow() {
    assert!(
        RenderOptions {
            width: 719,
            ..Default::default()
        }
        .validate(320)
        .is_err()
    );
    assert!(
        RenderOptions {
            width: 4097,
            ..Default::default()
        }
        .validate(320)
        .is_err()
    );
    assert!(
        RenderOptions {
            scale: f64::NAN,
            ..Default::default()
        }
        .validate(320)
        .is_err()
    );
    assert!(
        RenderOptions {
            scale: 4.1,
            ..Default::default()
        }
        .validate(320)
        .is_err()
    );
    assert!(RenderOptions::default().validate(100_000).is_err());
}

#[test]
fn options_accept_boundaries_and_return_physical_size() {
    let low_width = RenderOptions {
        width: 720,
        ..Default::default()
    }
    .validate(320)
    .unwrap();
    assert_eq!(low_width.width, 720);
    assert_eq!(low_width.height, 320);

    let high_width = RenderOptions {
        width: 4096,
        ..Default::default()
    }
    .validate(320)
    .unwrap();
    assert_eq!(high_width.width, 4096);
    assert_eq!(high_width.height, 320);

    let low_scale = RenderOptions {
        scale: 0.5,
        ..Default::default()
    }
    .validate(321)
    .unwrap();
    assert_eq!(low_scale.width, 600);
    assert_eq!(low_scale.height, 161);

    let high_scale = RenderOptions {
        scale: 4.0,
        width: 1000,
        ..Default::default()
    }
    .validate(1000)
    .unwrap();
    assert_eq!(high_scale.width, 4000);
    assert_eq!(high_scale.height, 4000);
}

#[test]
fn render_errors_have_stable_messages() {
    assert_eq!(
        RenderError::InvalidOption {
            option: "width",
            reason: "must be in 720..=4096",
        }
        .to_string(),
        "invalid render option width: must be in 720..=4096"
    );
    assert_eq!(
        RenderError::MissingTable {
            id: "returns.summary".to_string()
        }
        .to_string(),
        "missing report table returns.summary"
    );
    assert_eq!(
        RenderError::MissingColumn {
            table: "returns.summary".to_string(),
            column: "period".to_string(),
        }
        .to_string(),
        "missing column period in report table returns.summary"
    );
    assert_eq!(
        RenderError::InvalidColumnType {
            table: "returns.summary".to_string(),
            column: "period".to_string(),
            expected: "string",
        }
        .to_string(),
        "invalid column type for returns.summary.period: expected string"
    );
    assert_eq!(
        RenderError::InvalidContract {
            reason: "panel id is empty".to_string()
        }
        .to_string(),
        "invalid render contract: panel id is empty"
    );
    assert_eq!(
        RenderError::LayoutOverflow.to_string(),
        "render layout exceeds resource limits"
    );
    assert_eq!(
        RenderError::Allocation { bytes: 48_000_001 }.to_string(),
        "render buffer allocation failed for 48000001 bytes"
    );
    assert_eq!(
        RenderError::OutputTooLarge {
            format: "PNG",
            limit: 67_108_864
        }
        .to_string(),
        "rendered PNG output exceeds 67108864 bytes"
    );
    assert_eq!(
        RenderError::Backend {
            format: "SVG",
            message: "present failed".to_string()
        }
        .to_string(),
        "SVG backend failed: present failed"
    );
    assert_eq!(
        RenderError::PngEncoding {
            message: "bad chunk".to_string()
        }
        .to_string(),
        "PNG encoding failed: bad chunk"
    );
}

#[test]
fn font_assets_are_pinned_and_registration_is_idempotent() {
    assert_eq!(
        font_sha256(),
        "a3041811a78c361b1de50f953c805e0244951c21c5bd412f7232ef0d899af0da"
    );
    assert_eq!(font_bytes().len(), 17_772_300);
    assert!(font_license().contains("SIL OPEN FONT LICENSE Version 1.1"));

    ensure_font_registered().unwrap();
    ensure_font_registered().unwrap();
}
