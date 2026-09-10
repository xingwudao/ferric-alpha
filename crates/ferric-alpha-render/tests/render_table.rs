mod common;

use std::collections::HashMap;

use ferric_alpha::{
    ColumnRole, DisplayFormat, DisplayUnit, ReportColumn, ReportColumnData, ReportTable,
};
use ferric_alpha_render::{
    RenderOptions, RenderTheme, Theme, checked_padded_range, contrast_ratio, date_tick_indices,
    render_svg, render_table_panel_svg, series_color, theme_for,
};

#[test]
fn theme_palette_and_ranges_are_stable() {
    let light = theme_for(RenderTheme::Light);
    let dark = theme_for(RenderTheme::Dark);
    assert_eq!(light.background.hex(), "#ffffff");
    assert_eq!(light.text.hex(), "#1f2937");
    assert_eq!(light.grid.hex(), "#e5e7eb");
    assert_eq!(dark.background.hex(), "#111827");
    assert_eq!(dark.text.hex(), "#f9fafb");
    assert!(contrast_ratio(light.text, light.background) >= 4.5);
    assert!(contrast_ratio(dark.text, dark.background) >= 4.5);

    assert_eq!(series_color(0).hex(), "#2563eb");
    assert_eq!(series_color(7).hex(), "#2563eb");
    assert_eq!(
        Theme::series_color_for_key("1D").hex(),
        Theme::series_color_for_key("1D").hex()
    );
    assert_ne!(
        Theme::series_color_for_key("1D").hex(),
        Theme::series_color_for_key("3D").hex()
    );

    assert_eq!(
        checked_padded_range([Some(-2.0), None, Some(4.0)].into_iter(), false).unwrap(),
        Some((-2.6, 4.6))
    );
    assert_eq!(
        checked_padded_range([Some(0.0), Some(0.0)].into_iter(), true).unwrap(),
        Some((-1.0, 1.0))
    );
    assert_eq!(
        checked_padded_range([Some(5.0)].into_iter(), false).unwrap(),
        Some((4.5, 5.5))
    );
    assert_eq!(
        checked_padded_range([None].into_iter(), true).unwrap(),
        None
    );
    assert!(
        checked_padded_range([Some(f64::MAX), Some(f64::MAX / 2.0)].into_iter(), false).is_err()
    );

    assert_eq!(date_tick_indices(0, 5), Vec::<usize>::new());
    assert_eq!(date_tick_indices(3, 5), vec![0, 1, 2]);
    assert_eq!(date_tick_indices(10, 4), vec![0, 3, 6, 9]);
}

#[test]
fn table_panel_svg_contains_ordered_text_and_allocations() {
    let (_, report) = common::phase_05_report_cases()
        .into_iter()
        .find(|(name, _)| name == "summary")
        .unwrap();
    let plan = ferric_alpha_render::plan_report(
        &report,
        &RenderOptions {
            width: 720,
            ..RenderOptions::default()
        },
    )
    .unwrap();
    let panel = plan
        .panels
        .iter()
        .find(|panel| panel.id == "summary.returns-summary")
        .unwrap();
    let rendered =
        render_table_panel_svg(&report, panel, 720, theme_for(RenderTheme::Light)).unwrap();
    eprintln!(
        "summary.returns-summary panel SVG bytes: {}",
        rendered.svg.len()
    );

    assert!(rendered.svg.contains("id=\"summary.returns-summary\""));
    assert!(rendered.svg.contains("Returns Summary"));
    assert!(rendered.svg.contains("Metric"));
    assert!(rendered.svg.contains("Period"));
    assert!(rendered.svg.contains("Value"));
    assert!(rendered.svg.contains("435.029499%"));
    roxmltree::Document::parse(&rendered.svg).unwrap();

    let text_nodes = roxmltree::Document::parse(&rendered.svg)
        .unwrap()
        .descendants()
        .filter(|node| node.has_tag_name("text"))
        .count();
    assert_eq!(text_nodes, rendered.text_allocations.len());
    assert!(
        rendered
            .text_allocations
            .iter()
            .all(|allocation| allocation.measured_width
                >= allocation.raw_width.saturating_mul(5) / 4)
    );
    assert!(rendered.text_allocations.iter().all(|allocation| {
        allocation.x + allocation.width <= 720 && allocation.y + allocation.height <= panel.height
    }));
}

#[test]
fn table_panel_handles_nulls_unicode_xml_and_long_labels() {
    let table = special_table();
    let rendered = ferric_alpha_render::render_table_svg(
        "fixture.special",
        "特殊 <Table> & Values",
        &[&table],
        260,
        260,
        theme_for(RenderTheme::Light),
    )
    .unwrap();

    assert!(rendered.svg.contains("特殊"));
    assert!(rendered.svg.contains("&lt;Table&gt;"));
    assert!(rendered.svg.contains("&amp;"));
    assert!(rendered.svg.contains("—"));
    assert!(rendered.svg.contains("12.34%"));
    assert!(rendered.svg.contains("…"));
    roxmltree::Document::parse(&rendered.svg).unwrap();
    assert!(
        rendered
            .text_allocations
            .iter()
            .all(|allocation| allocation.x + allocation.width <= 260)
    );
}

#[test]
fn empty_table_renders_stable_empty_body_and_bytes_repeat() {
    let table = empty_table();
    let first = ferric_alpha_render::render_table_svg(
        "fixture.empty",
        "Empty Table",
        &[&table],
        720,
        180,
        theme_for(RenderTheme::Light),
    )
    .unwrap();
    let second = ferric_alpha_render::render_table_svg(
        "fixture.empty",
        "Empty Table",
        &[&table],
        720,
        180,
        theme_for(RenderTheme::Light),
    )
    .unwrap();

    assert_eq!(first.svg, second.svg);
    assert_eq!(first.text_allocations, second.text_allocations);
    assert!(first.svg.contains("No rows"));
    roxmltree::Document::parse(&first.svg).unwrap();
}

#[test]
fn initial_report_svg_renders_table_panels_deterministically() {
    let (_, report) = common::phase_05_report_cases()
        .into_iter()
        .find(|(name, _)| name == "summary")
        .unwrap();
    let options = RenderOptions {
        width: 1200,
        ..RenderOptions::default()
    };
    let first = render_svg(&report, &options).unwrap();
    let second = render_svg(&report, &options).unwrap();
    eprintln!("summary report SVG bytes: {}", first.len());
    assert_eq!(first, second);
    assert!(first.contains("<svg"));
    assert!(first.contains("summary.quantile-statistics"));
    assert!(first.contains("summary.returns-summary"));
    assert!(first.contains("summary.information-summary"));
    roxmltree::Document::parse(&first).unwrap();

    let ids = roxmltree::Document::parse(&first)
        .unwrap()
        .descendants()
        .filter_map(|node| node.attribute("id"))
        .fold(HashMap::<String, usize>::new(), |mut counts, id| {
            *counts.entry(id.to_string()).or_default() += 1;
            counts
        });
    assert_eq!(ids.get("summary.returns-summary"), Some(&1));
}

fn special_table() -> ReportTable {
    ReportTable::new(
        "fixture.special",
        "Special",
        vec!["name".to_string()],
        vec![
            ReportColumn::new(
                "name",
                "中文 <Name> & Label",
                ColumnRole::Dimension,
                None,
                ReportColumnData::String(vec![
                    Some("a very very very very very very long label".to_string()),
                    Some("短 & <safe>".to_string()),
                ]),
            ),
            ReportColumn::new(
                "value",
                "Value",
                ColumnRole::Measure,
                Some(DisplayFormat::new(DisplayUnit::DecimalReturn, 2)),
                ReportColumnData::Float64(vec![Some(0.1234), None]),
            ),
        ],
    )
    .unwrap()
}

fn empty_table() -> ReportTable {
    ReportTable::new(
        "fixture.empty",
        "Empty",
        vec![],
        vec![ReportColumn::new(
            "name",
            "Name",
            ColumnRole::Dimension,
            None,
            ReportColumnData::String(vec![]),
        )],
    )
    .unwrap()
}
