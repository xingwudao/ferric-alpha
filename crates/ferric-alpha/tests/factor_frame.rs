use ferric_alpha::{FerricAlphaError, validate_factor_frame};
use polars::prelude::*;

const JAN_1_2024_MS: i64 = 1_704_067_200_000;
const JAN_2_2024_MS: i64 = 1_704_153_600_000;

fn datetime_column(values: &[Option<i64>]) -> Column {
    Series::new("date".into(), values)
        .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
        .expect("test dates should cast to datetime")
        .into()
}

fn factor_frame(
    dates: &[Option<i64>],
    assets: &[Option<&str>],
    factors: &[Option<f64>],
) -> DataFrame {
    DataFrame::new(
        dates.len(),
        vec![
            datetime_column(dates),
            Series::new("asset".into(), assets).into(),
            Series::new("factor".into(), factors).into(),
        ],
    )
    .expect("test frame should be valid")
}

#[test]
fn validation_sorts_by_date_then_asset() {
    let frame = factor_frame(
        &[
            Some(JAN_2_2024_MS),
            Some(JAN_1_2024_MS),
            Some(JAN_1_2024_MS),
        ],
        &[Some("B"), Some("B"), Some("A")],
        &[Some(3.0), Some(2.0), Some(1.0)],
    );

    let result = validate_factor_frame(&frame).expect("valid frame should pass");

    let assets: Vec<_> = result
        .column("asset")
        .expect("asset should exist")
        .str()
        .expect("asset should be a string")
        .iter()
        .collect();
    assert_eq!(assets, vec![Some("A"), Some("B"), Some("B")]);

    let factors: Vec<_> = result
        .column("factor")
        .expect("factor should exist")
        .f64()
        .expect("factor should be float64")
        .iter()
        .collect();
    assert_eq!(factors, vec![Some(1.0), Some(2.0), Some(3.0)]);
}

#[test]
fn validation_accepts_optional_string_group() {
    let mut frame = factor_frame(&[Some(JAN_1_2024_MS)], &[Some("A")], &[Some(1.0)]);
    frame
        .with_column(Series::new("group".into(), &[Some("technology")]).into())
        .expect("group should be added");

    let result = validate_factor_frame(&frame).expect("string group should pass");

    assert_eq!(result.column("group").unwrap().dtype(), &DataType::String);
}

#[test]
fn validation_preserves_null_factor_values() {
    let frame = factor_frame(
        &[Some(JAN_1_2024_MS), Some(JAN_2_2024_MS)],
        &[Some("A"), Some("A")],
        &[None, Some(1.0)],
    );

    let result = validate_factor_frame(&frame).expect("null factors should pass");

    assert_eq!(result.column("factor").unwrap().null_count(), 1);
}

#[test]
fn validation_rejects_missing_factor_column() {
    let frame = DataFrame::new(
        1,
        vec![
            datetime_column(&[Some(JAN_1_2024_MS)]),
            Series::new("asset".into(), &["A"]).into(),
        ],
    )
    .expect("test frame should be valid");

    let error = validate_factor_frame(&frame).expect_err("missing factor should fail");

    assert!(matches!(error, FerricAlphaError::MissingColumn("factor")));
}

#[test]
fn validation_rejects_missing_date_column() {
    let frame = DataFrame::new(
        1,
        vec![
            Series::new("asset".into(), &["A"]).into(),
            Series::new("factor".into(), &[1.0]).into(),
        ],
    )
    .expect("test frame should be valid");

    let error = validate_factor_frame(&frame).expect_err("missing date should fail");

    assert!(matches!(error, FerricAlphaError::MissingColumn("date")));
}

#[test]
fn validation_rejects_missing_asset_column() {
    let frame = DataFrame::new(
        1,
        vec![
            datetime_column(&[Some(JAN_1_2024_MS)]),
            Series::new("factor".into(), &[1.0]).into(),
        ],
    )
    .expect("test frame should be valid");

    let error = validate_factor_frame(&frame).expect_err("missing asset should fail");

    assert!(matches!(error, FerricAlphaError::MissingColumn("asset")));
}

#[test]
fn validation_rejects_duplicate_date_asset_key() {
    let frame = factor_frame(
        &[Some(JAN_1_2024_MS), Some(JAN_1_2024_MS)],
        &[Some("A"), Some("A")],
        &[Some(1.0), Some(2.0)],
    );

    let error = validate_factor_frame(&frame).expect_err("duplicate key should fail");

    assert!(matches!(error, FerricAlphaError::DuplicateKey));
}

#[test]
fn validation_rejects_null_date_key() {
    let frame = factor_frame(&[None], &[Some("A")], &[Some(1.0)]);

    let error = validate_factor_frame(&frame).expect_err("null date should fail");

    assert!(matches!(error, FerricAlphaError::NullKey));
}

#[test]
fn validation_rejects_null_asset_key() {
    let frame = factor_frame(&[Some(JAN_1_2024_MS)], &[None], &[Some(1.0)]);

    let error = validate_factor_frame(&frame).expect_err("null asset should fail");

    assert!(matches!(error, FerricAlphaError::NullKey));
}

#[test]
fn validation_rejects_non_datetime_date_column() {
    let frame = DataFrame::new(
        1,
        vec![
            Series::new("date".into(), &[JAN_1_2024_MS]).into(),
            Series::new("asset".into(), &["A"]).into(),
            Series::new("factor".into(), &[1.0]).into(),
        ],
    )
    .expect("test frame should be valid");

    let error = validate_factor_frame(&frame).expect_err("integer date should fail");

    assert!(matches!(
        error,
        FerricAlphaError::InvalidDtype { column: "date", .. }
    ));
}

#[test]
fn validation_rejects_non_string_asset_column() {
    let frame = DataFrame::new(
        1,
        vec![
            datetime_column(&[Some(JAN_1_2024_MS)]),
            Series::new("asset".into(), &[1_i64]).into(),
            Series::new("factor".into(), &[1.0]).into(),
        ],
    )
    .expect("test frame should be valid");

    let error = validate_factor_frame(&frame).expect_err("integer asset should fail");

    assert!(matches!(
        error,
        FerricAlphaError::InvalidDtype {
            column: "asset",
            ..
        }
    ));
}

#[test]
fn validation_rejects_non_float64_factor_column() {
    let frame = DataFrame::new(
        1,
        vec![
            datetime_column(&[Some(JAN_1_2024_MS)]),
            Series::new("asset".into(), &["A"]).into(),
            Series::new("factor".into(), &[1_i64]).into(),
        ],
    )
    .expect("test frame should be valid");

    let error = validate_factor_frame(&frame).expect_err("integer factor should fail");

    assert!(matches!(
        error,
        FerricAlphaError::InvalidDtype {
            column: "factor",
            ..
        }
    ));
}

#[test]
fn validation_rejects_non_string_group_column() {
    let mut frame = factor_frame(&[Some(JAN_1_2024_MS)], &[Some("A")], &[Some(1.0)]);
    frame
        .with_column(Series::new("group".into(), &[1_i64]).into())
        .expect("group should be added");

    let error = validate_factor_frame(&frame).expect_err("integer group should fail");

    assert!(matches!(
        error,
        FerricAlphaError::InvalidDtype {
            column: "group",
            ..
        }
    ));
}
