mod common;

use std::path::Path;

use ferric_alpha_render::{RenderOptions, plan_report, render_html, render_png, render_svg};

#[test]
fn phase_05_golden_artifacts_match_current_renderer() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("tests/golden/phase-05");
    let manifest = std::fs::read_to_string(root.join("manifest.json")).unwrap();
    assert!(manifest.contains("ferric-alpha.render-plan/v1"));

    let cases = common::phase_05_report_cases();
    let plans = cases
        .iter()
        .map(|(name, report)| {
            let plan = plan_report(report, &RenderOptions::default()).unwrap();
            serde_json::json!({"name": name, "plan": serde_json::from_str::<serde_json::Value>(&plan.to_json().unwrap()).unwrap()})
        })
        .collect::<Vec<_>>();
    let expected_plans = std::fs::read_to_string(root.join("render_plans.json")).unwrap();
    assert_eq!(
        serde_json::to_value(&plans).unwrap(),
        serde_json::from_str::<serde_json::Value>(&expected_plans).unwrap()
    );

    let (_, summary) = cases.iter().find(|(name, _)| name == "summary").unwrap();
    assert_eq!(
        render_svg(summary, &RenderOptions::default()).unwrap(),
        std::fs::read_to_string(root.join("report-small.svg")).unwrap()
    );
    assert_eq!(
        render_html(summary, &RenderOptions::default()).unwrap(),
        std::fs::read_to_string(root.join("report-small.html")).unwrap()
    );
    assert!(
        render_png(summary, &RenderOptions::default())
            .unwrap()
            .len()
            > 1024
    );

    let roles = std::fs::read_to_string(root.join("semantic_roles.json")).unwrap();
    assert!(roles.contains("mean_return_by_quantile"));
    let pixels = std::fs::read_to_string(root.join("report-small-pixels.json")).unwrap();
    assert!(pixels.contains("non_background_ratio"));
}
