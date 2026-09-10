use ferric_alpha::TearSheetData;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct ReportCase {
    pub name: String,
    pub canonical_json: String,
}

pub fn phase_05_report_cases() -> Vec<(String, TearSheetData)> {
    let fixture = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join("tests/golden/phase-05/report_cases.json"),
    )
    .expect("phase-05 report cases should exist");
    let cases = serde_json::from_str::<Vec<ReportCase>>(&fixture)
        .expect("phase-05 report cases should be valid JSON");
    cases
        .into_iter()
        .map(|case| {
            let report = TearSheetData::from_json(&case.canonical_json)
                .unwrap_or_else(|error| panic!("invalid report case {}: {error:?}", case.name));
            (case.name, report)
        })
        .collect()
}
