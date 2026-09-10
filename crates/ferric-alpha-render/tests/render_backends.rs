mod common;

use std::io::Write;

use ferric_alpha_render::{
    LimitedVecWriter, MAX_PNG_BYTES, RenderOptions, render_html, render_html_fragment, render_png,
    render_svg, write_html, write_png, write_svg,
};
use serde_json::Value;

#[test]
fn svg_png_and_html_render_for_all_report_cases() {
    for (name, report) in common::phase_05_report_cases() {
        let options = RenderOptions {
            width: 720,
            ..RenderOptions::default()
        };
        let svg1 = render_svg(&report, &options).unwrap();
        let svg2 = render_svg(&report, &options).unwrap();
        assert_eq!(svg1, svg2, "{name}");
        assert!(svg1.starts_with("<svg"));
        assert!(svg1.contains("data-role=\"plot-body\"") || svg1.contains("<table"));
        assert!(!svg1.contains("<script"));
        assert!(!svg1.contains("href=\"http://"));
        assert!(!svg1.contains("href=\"https://"));
        roxmltree::Document::parse(&svg1).unwrap();

        let png = render_png(&report, &options).unwrap();
        assert!(png.starts_with(b"\x89PNG\r\n\x1a\n"));
        let decoder = png::Decoder::new(std::io::Cursor::new(&png));
        let reader = decoder.read_info().unwrap();
        let info = reader.info();
        assert_eq!(info.width, 720);
        assert!(info.height > 0);
        assert_eq!(info.color_type, png::ColorType::Rgb);
        assert_eq!(info.bit_depth, png::BitDepth::Eight);

        let html = render_html(&report, &options).unwrap();
        assert!(html.starts_with("<!doctype html>"));
        assert!(html.contains("<div class=\"ferric-alpha-report\""));
        assert!(html.contains("<table"));
        assert!(html.contains("<svg"));
        assert!(!html.contains("<script"));
        assert!(!html.contains("href=\"http://"));
        assert!(!html.contains("href=\"https://"));

        let fragment = render_html_fragment(&report, &options).unwrap();
        assert!(fragment.starts_with("<div class=\"ferric-alpha-report\""));
        assert!(!fragment.contains("<html"));
        assert!(!fragment.contains("<body"));
    }
}

#[test]
fn scaled_png_dimensions_are_physical_pixels() {
    let (_, report) = common::phase_05_report_cases()
        .into_iter()
        .find(|(name, _)| name == "summary")
        .unwrap();
    let options = RenderOptions {
        width: 1200,
        scale: 2.0,
        ..RenderOptions::default()
    };
    let png = render_png(&report, &options).unwrap();
    let decoder = png::Decoder::new(std::io::Cursor::new(&png));
    let reader = decoder.read_info().unwrap();
    assert_eq!(reader.info().width, 2400);
}

#[test]
fn png_render_changes_when_report_numbers_change() {
    let (_, report) = common::phase_05_report_cases()
        .into_iter()
        .find(|(name, _)| name == "returns")
        .unwrap();
    let options = RenderOptions {
        width: 720,
        ..RenderOptions::default()
    };
    let baseline = render_png(&report, &options).unwrap();
    let mutated = multiply_report_float_columns(&report, -3.0);
    let changed = render_png(&mutated, &options).unwrap();
    assert_ne!(baseline, changed);
}

#[test]
fn file_helpers_write_same_bytes_as_memory_renderers() {
    let (_, report) = common::phase_05_report_cases()
        .into_iter()
        .find(|(name, _)| name == "summary")
        .unwrap();
    let dir = tempfile::tempdir().unwrap();
    let svg_path = dir.path().join("report.svg");
    let png_path = dir.path().join("report.png");
    let html_path = dir.path().join("report.html");
    let options = RenderOptions::default();

    write_svg(&report, &options, &svg_path).unwrap();
    write_png(&report, &options, &png_path).unwrap();
    write_html(&report, &options, &html_path).unwrap();

    assert_eq!(
        std::fs::read_to_string(svg_path).unwrap(),
        render_svg(&report, &options).unwrap()
    );
    assert_eq!(
        std::fs::read(png_path).unwrap(),
        render_png(&report, &options).unwrap()
    );
    assert_eq!(
        std::fs::read_to_string(html_path).unwrap(),
        render_html(&report, &options).unwrap()
    );
}

#[test]
fn limited_vec_writer_rejects_growth_past_limit() {
    let mut writer = LimitedVecWriter::new(4);
    writer.write_all(&[1, 2, 3, 4]).unwrap();
    assert_eq!(writer.finish().unwrap(), vec![1, 2, 3, 4]);

    let mut writer = LimitedVecWriter::new(4);
    assert!(writer.write_all(&[1, 2, 3, 4, 5]).is_err());
    let err = writer.finish().unwrap_err();
    assert!(matches!(
        err,
        ferric_alpha_render::RenderError::OutputTooLarge { .. }
    ));

    assert_eq!(MAX_PNG_BYTES, 64 * 1024 * 1024);
}

fn multiply_report_float_columns(
    report: &ferric_alpha::TearSheetData,
    factor: f64,
) -> ferric_alpha::TearSheetData {
    let mut value = serde_json::from_str::<Value>(&report.to_json().unwrap()).unwrap();
    multiply_float64_values(&mut value, factor);
    ferric_alpha::TearSheetData::from_json(&serde_json::to_string(&value).unwrap()).unwrap()
}

fn multiply_float64_values(value: &mut Value, factor: f64) {
    match value {
        Value::Object(map) => {
            if let Some(Value::Array(values)) = map.get_mut("float64") {
                for value in values {
                    if let Some(number) = value.as_f64()
                        && let Some(next) = serde_json::Number::from_f64(number * factor)
                    {
                        *value = Value::Number(next);
                    }
                }
            }
            for value in map.values_mut() {
                multiply_float64_values(value, factor);
            }
        }
        Value::Array(values) => {
            for value in values {
                multiply_float64_values(value, factor);
            }
        }
        _ => {}
    }
}
