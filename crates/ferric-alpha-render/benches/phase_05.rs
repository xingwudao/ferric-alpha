use criterion::{Criterion, criterion_group, criterion_main};
use ferric_alpha::TearSheetData;
use ferric_alpha_render::{RenderOptions, plan_report, render_html, render_png, render_svg};

fn load_full_report() -> TearSheetData {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("tests/golden/phase-05/report_cases.json");
    let cases: Vec<serde_json::Value> =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let json = cases.iter().find(|case| case["name"] == "full").unwrap()["canonical_json"]
        .as_str()
        .unwrap();
    TearSheetData::from_json(json).unwrap()
}

fn phase_05_benchmarks(c: &mut Criterion) {
    let report = load_full_report();
    let options = RenderOptions::default();
    c.bench_function("render-plan/full", |b| {
        b.iter(|| plan_report(&report, &options).unwrap())
    });
    c.bench_function("render-svg/full", |b| {
        b.iter(|| render_svg(&report, &options).unwrap())
    });
    c.bench_function("render-png/full", |b| {
        b.iter(|| render_png(&report, &options).unwrap())
    });
    c.bench_function("render-html/full", |b| {
        b.iter(|| render_html(&report, &options).unwrap())
    });
}

criterion_group! {
    name = benches;
    config = Criterion::default().without_plots();
    targets = phase_05_benchmarks
}
criterion_main!(benches);
