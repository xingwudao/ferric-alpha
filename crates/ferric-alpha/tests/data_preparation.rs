use ferric_alpha::{
    CleanOptions, FerricAlphaError, ForwardReturnsOptions, Period, QuantizeOptions,
    TradingCalendar, compute_forward_returns, compute_forward_returns_with_options,
    get_clean_factor_and_forward_returns, quantize_factor,
};
use polars::prelude::*;

const JAN_1: i64 = 1_704_067_200_000;
const JAN_2: i64 = 1_704_153_600_000;
const JAN_3: i64 = 1_704_240_000_000;
const JAN_4: i64 = 1_704_326_400_000;
const JAN_5: i64 = 1_704_412_800_000;
const JAN_8: i64 = 1_704_672_000_000;

fn datetime_column(values: &[i64]) -> Column {
    Series::new("date".into(), values)
        .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
        .expect("test dates should cast to datetime")
        .into()
}

fn utc_datetime_column(values: &[i64]) -> Column {
    Series::new("date".into(), values)
        .into_datetime(
            TimeUnit::Milliseconds,
            TimeZone::opt_try_new(Some("Asia/Shanghai")).expect("timezone should be valid"),
        )
        .into_column()
}

fn factor_frame(dates: &[i64], assets: &[&str], factors: &[Option<f64>]) -> DataFrame {
    DataFrame::new(
        dates.len(),
        vec![
            datetime_column(dates),
            Series::new("asset".into(), assets).into(),
            Series::new("factor".into(), factors).into(),
        ],
    )
    .expect("test factor frame should be valid")
}

fn grouped_factor_frame(
    dates: &[i64],
    assets: &[&str],
    groups: &[&str],
    factors: &[Option<f64>],
) -> DataFrame {
    let mut frame = factor_frame(dates, assets, factors);
    frame
        .with_column(Series::new("group".into(), groups).into())
        .expect("group column should be added");
    frame
}

fn price_frame(dates: &[i64], assets: &[&str], prices: &[Option<f64>]) -> DataFrame {
    DataFrame::new(
        dates.len(),
        vec![
            datetime_column(dates),
            Series::new("asset".into(), assets).into(),
            Series::new("price".into(), prices).into(),
        ],
    )
    .expect("test price frame should be valid")
}

fn asset_quantile_pairs(frame: &DataFrame) -> Vec<(String, Option<u32>)> {
    let assets = frame.column("asset").unwrap().str().unwrap();
    let quantiles = frame.column("factor_quantile").unwrap().u32().unwrap();
    assets
        .iter()
        .zip(quantiles.iter())
        .map(|(asset, quantile)| {
            (
                asset.expect("asset should not be null").to_owned(),
                quantile,
            )
        })
        .collect()
}

#[test]
fn period_rejects_zero_counts() {
    let error = Period::new(0).expect_err("zero observation period should fail");

    assert!(matches!(
        error,
        FerricAlphaError::InvalidPeriod { count: 0 }
    ));
}

#[test]
fn calendar_rejects_unsorted_and_duplicate_sessions() {
    let unsorted =
        TradingCalendar::new(vec![JAN_2, JAN_1], None).expect_err("unsorted sessions should fail");
    let duplicate =
        TradingCalendar::new(vec![JAN_1, JAN_1], None).expect_err("duplicate sessions should fail");

    assert!(matches!(unsorted, FerricAlphaError::UnsortedSessions));
    assert!(matches!(duplicate, FerricAlphaError::DuplicateSession));
}

#[test]
fn calendar_offsets_by_observed_sessions() {
    let calendar = TradingCalendar::new(vec![JAN_1, JAN_2, JAN_3, JAN_4, JAN_5, JAN_8], None)
        .expect("calendar should be valid");

    assert_eq!(calendar.offset(JAN_1, Period::new(5).unwrap()), Some(JAN_8));
    assert_eq!(calendar.offset(JAN_5, Period::new(1).unwrap()), Some(JAN_8));
    assert_eq!(calendar.offset(JAN_8, Period::new(1).unwrap()), None);
}

#[test]
fn forward_returns_compute_requested_observation_periods() {
    let factor = factor_frame(
        &[JAN_1, JAN_1, JAN_2],
        &["B", "A", "A"],
        &[Some(2.0), Some(1.0), Some(1.5)],
    );
    let prices = price_frame(
        &[JAN_1, JAN_2, JAN_3, JAN_1, JAN_2, JAN_3],
        &["A", "A", "A", "B", "B", "B"],
        &[
            Some(100.0),
            Some(110.0),
            Some(121.0),
            Some(50.0),
            Some(40.0),
            Some(60.0),
        ],
    );

    let result = compute_forward_returns(&factor, &prices, &[Period::new(1).unwrap()])
        .expect("forward returns should compute");

    let assets: Vec<_> = result
        .column("asset")
        .unwrap()
        .str()
        .unwrap()
        .iter()
        .collect();
    let returns: Vec<_> = result
        .column("forward_return_1D")
        .unwrap()
        .f64()
        .unwrap()
        .iter()
        .collect();
    assert_eq!(assets, vec![Some("A"), Some("B"), Some("A")]);
    assert_eq!(returns, vec![Some(0.1), Some(-0.2), Some(0.1)]);
}

#[test]
fn alphalens_forward_returns_ignore_price_history_before_factor_dates() {
    let factor = factor_frame(
        &[JAN_1, JAN_1, JAN_2, JAN_2, JAN_3, JAN_3],
        &["A", "B", "A", "B", "A", "B"],
        &[
            Some(1.0),
            Some(1.0),
            Some(1.0),
            Some(2.0),
            Some(2.0),
            Some(1.0),
        ],
    );
    let dec_29_2023 = JAN_1 - 3 * 86_400_000;
    let dec_30_2023 = JAN_1 - 2 * 86_400_000;
    let dec_31_2023 = JAN_1 - 86_400_000;
    let prices = price_frame(
        &[
            dec_29_2023,
            dec_29_2023,
            dec_30_2023,
            dec_30_2023,
            dec_31_2023,
            dec_31_2023,
            JAN_1,
            JAN_1,
            JAN_2,
            JAN_2,
            JAN_3,
            JAN_3,
        ],
        &["A", "B", "A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
        &[
            None,
            None,
            None,
            None,
            None,
            None,
            Some(1.0),
            Some(1.0),
            Some(1.0),
            Some(2.0),
            Some(2.0),
            Some(1.0),
        ],
    );

    let result = compute_forward_returns(
        &factor,
        &prices,
        &[Period::new(1).unwrap(), Period::new(2).unwrap()],
    )
    .unwrap();
    let one_day = result.column("forward_return_1D").unwrap().f64().unwrap();
    let two_day = result.column("forward_return_2D").unwrap().f64().unwrap();

    assert_eq!(
        one_day.iter().collect::<Vec<_>>(),
        vec![Some(0.0), Some(1.0), Some(1.0), Some(-0.5), None, None]
    );
    assert_eq!(
        two_day.iter().collect::<Vec<_>>(),
        vec![Some(1.0), Some(0.0), None, None, None, None]
    );
}

#[test]
fn alphalens_forward_returns_can_compute_non_cumulative_period_returns() {
    let factor = factor_frame(
        &[JAN_1, JAN_1, JAN_2],
        &["A", "B", "A"],
        &[Some(1.0), Some(1.0), Some(1.0)],
    );
    let prices = price_frame(
        &[JAN_1, JAN_2, JAN_3, JAN_1, JAN_2, JAN_3],
        &["A", "A", "A", "B", "B", "B"],
        &[
            Some(100.0),
            Some(110.0),
            Some(121.0),
            Some(50.0),
            Some(40.0),
            Some(60.0),
        ],
    );

    let result = compute_forward_returns_with_options(
        &factor,
        &prices,
        &[Period::new(1).unwrap(), Period::new(2).unwrap()],
        ForwardReturnsOptions {
            cumulative_returns: false,
            filter_zscore: None,
        },
    )
    .unwrap();
    let one_day = result.column("forward_return_1D").unwrap().f64().unwrap();
    let two_day = result.column("forward_return_2D").unwrap().f64().unwrap();

    assert_eq!(
        one_day.iter().collect::<Vec<_>>(),
        vec![Some(0.1), Some(-0.2), Some(0.1)]
    );
    assert_eq!(
        two_day.iter().collect::<Vec<_>>(),
        vec![Some(0.1), Some(0.5), None]
    );
}

#[test]
fn alphalens_forward_returns_infer_intraday_period_labels_from_price_sessions() {
    let one_hour = 60 * 60 * 1_000;
    let three_hours = 3 * one_hour;
    let factor = factor_frame(
        &[JAN_2 + 9 * one_hour + 30 * 60 * 1_000],
        &["A"],
        &[Some(1.0)],
    );
    let open = JAN_2 + 9 * one_hour + 30 * 60 * 1_000;
    let prices = price_frame(
        &[
            open,
            open + one_hour,
            open + three_hours,
            JAN_3 + 9 * one_hour + 30 * 60 * 1_000,
        ],
        &["A", "A", "A", "A"],
        &[Some(100.0), Some(101.0), Some(98.0), Some(110.0)],
    );

    let result = compute_forward_returns(
        &factor,
        &prices,
        &[
            Period::new(1).unwrap(),
            Period::new(2).unwrap(),
            Period::new(3).unwrap(),
        ],
    )
    .unwrap();

    assert_eq!(
        result
            .get_column_names()
            .iter()
            .map(|name| name.as_str())
            .collect::<Vec<_>>(),
        vec![
            "date",
            "asset",
            "forward_return_1h",
            "forward_return_3h",
            "forward_return_1D"
        ]
    );
}

#[test]
fn forward_returns_support_multiple_observation_periods() {
    let factor = factor_frame(&[JAN_1], &["A"], &[Some(1.0)]);
    let prices = price_frame(
        &[JAN_1, JAN_2, JAN_3, JAN_4, JAN_5, JAN_8],
        &["A", "A", "A", "A", "A", "A"],
        &[
            Some(100.0),
            Some(110.0),
            Some(121.0),
            Some(133.1),
            Some(146.41),
            Some(161.051),
        ],
    );

    let result = compute_forward_returns(
        &factor,
        &prices,
        &[
            Period::new(1).unwrap(),
            Period::new(2).unwrap(),
            Period::new(5).unwrap(),
        ],
    )
    .expect("multiple forward periods should compute");

    assert_eq!(
        result
            .column("forward_return_1D")
            .unwrap()
            .f64()
            .unwrap()
            .get(0),
        Some(0.1)
    );
    assert_eq!(
        result
            .column("forward_return_2D")
            .unwrap()
            .f64()
            .unwrap()
            .get(0),
        Some(0.21)
    );
    assert_eq!(
        result
            .column("forward_return_5D")
            .unwrap()
            .f64()
            .unwrap()
            .get(0),
        Some(0.61051)
    );
}

#[test]
fn forward_returns_reject_duplicate_periods_and_timezone_mismatch() {
    let factor = factor_frame(&[JAN_1], &["A"], &[Some(1.0)]);
    let prices = price_frame(&[JAN_1, JAN_2], &["A", "A"], &[Some(100.0), Some(110.0)]);
    let duplicate_periods = compute_forward_returns(
        &factor,
        &prices,
        &[Period::new(1).unwrap(), Period::new(1).unwrap()],
    )
    .expect_err("duplicate periods should fail");

    let tz_factor = DataFrame::new(
        1,
        vec![
            utc_datetime_column(&[JAN_1]),
            Series::new("asset".into(), &["A"]).into(),
            Series::new("factor".into(), &[1.0]).into(),
        ],
    )
    .expect("timezone test frame should be valid");
    let timezone_mismatch =
        compute_forward_returns(&tz_factor, &prices, &[Period::new(1).unwrap()])
            .expect_err("timezone mismatch should fail");

    assert!(matches!(
        duplicate_periods,
        FerricAlphaError::DuplicatePeriod
    ));
    assert!(matches!(
        timezone_mismatch,
        FerricAlphaError::TimezoneMismatch
    ));
}

#[test]
fn forward_returns_preserve_input_timezone() {
    let factor = DataFrame::new(
        1,
        vec![
            utc_datetime_column(&[JAN_1]),
            Series::new("asset".into(), &["A"]).into(),
            Series::new("factor".into(), &[1.0]).into(),
        ],
    )
    .expect("timezone test factor should be valid");
    let prices = DataFrame::new(
        2,
        vec![
            utc_datetime_column(&[JAN_1, JAN_2]),
            Series::new("asset".into(), &["A", "A"]).into(),
            Series::new("price".into(), &[100.0, 110.0]).into(),
        ],
    )
    .expect("timezone test prices should be valid");

    let result = compute_forward_returns(&factor, &prices, &[Period::new(1).unwrap()])
        .expect("timezone-preserving returns should compute");

    assert_eq!(
        result.column("date").unwrap().dtype(),
        factor.column("date").unwrap().dtype()
    );
}

#[test]
fn forward_returns_leave_missing_terminal_prices_null() {
    let factor = factor_frame(&[JAN_2], &["A"], &[Some(1.0)]);
    let prices = price_frame(&[JAN_1, JAN_2], &["A", "A"], &[Some(100.0), Some(110.0)]);

    let result = compute_forward_returns(&factor, &prices, &[Period::new(1).unwrap()])
        .expect("missing terminal price should not reject the factor row");

    assert_eq!(
        result
            .column("forward_return_1D")
            .unwrap()
            .f64()
            .unwrap()
            .get(0),
        None
    );
}

#[test]
fn forward_returns_leave_missing_asset_prices_null() {
    let factor = factor_frame(&[JAN_1], &["B"], &[Some(1.0)]);
    let prices = price_frame(&[JAN_1, JAN_2], &["A", "A"], &[Some(100.0), Some(110.0)]);

    let result = compute_forward_returns(&factor, &prices, &[Period::new(1).unwrap()])
        .expect("missing asset prices should not reject the factor row");

    assert_eq!(
        result
            .column("forward_return_1D")
            .unwrap()
            .f64()
            .unwrap()
            .get(0),
        None
    );
}

#[test]
fn forward_returns_marks_null_price_windows_null_and_rejects_non_finite_prices() {
    let factor = factor_frame(&[JAN_1], &["A"], &[Some(1.0)]);
    let null_prices = price_frame(&[JAN_1, JAN_2], &["A", "A"], &[Some(100.0), None]);
    let nan_prices = price_frame(&[JAN_1, JAN_2], &["A", "A"], &[Some(100.0), Some(f64::NAN)]);

    let null_result = compute_forward_returns(&factor, &null_prices, &[Period::new(1).unwrap()])
        .expect("null terminal price should produce null forward return");
    let nan_error = compute_forward_returns(&factor, &nan_prices, &[Period::new(1).unwrap()])
        .expect_err("NaN prices should fail");

    assert_eq!(
        null_result
            .column("forward_return_1D")
            .unwrap()
            .f64()
            .unwrap()
            .get(0),
        None
    );
    assert!(matches!(nan_error, FerricAlphaError::InvalidPrice));
}

#[test]
fn forward_returns_use_observed_sparse_sessions() {
    let factor = factor_frame(&[JAN_5], &["A"], &[Some(1.0)]);
    let prices = price_frame(&[JAN_5, JAN_8], &["A", "A"], &[Some(100.0), Some(105.0)]);

    let result = compute_forward_returns(&factor, &prices, &[Period::new(1).unwrap()])
        .expect("sparse observed calendar should compute");

    assert_eq!(
        result
            .column("forward_return_1D")
            .unwrap()
            .f64()
            .unwrap()
            .get(0),
        Some(0.05)
    );
}

#[test]
fn forward_returns_reject_duplicate_price_keys_and_non_positive_prices() {
    let factor = factor_frame(&[JAN_1], &["A"], &[Some(1.0)]);
    let duplicate_prices = price_frame(
        &[JAN_1, JAN_1, JAN_2],
        &["A", "A", "A"],
        &[Some(100.0), Some(101.0), Some(110.0)],
    );
    let non_positive_prices = price_frame(&[JAN_1, JAN_2], &["A", "A"], &[Some(100.0), Some(0.0)]);

    let duplicate_error =
        compute_forward_returns(&factor, &duplicate_prices, &[Period::new(1).unwrap()])
            .expect_err("duplicate prices should fail");
    let price_error =
        compute_forward_returns(&factor, &non_positive_prices, &[Period::new(1).unwrap()])
            .expect_err("non-positive prices should fail");

    assert!(matches!(duplicate_error, FerricAlphaError::DuplicateKey));
    assert!(matches!(price_error, FerricAlphaError::InvalidPrice));
}

#[test]
fn quantize_factor_property_range_and_partition_completeness() {
    let factor = factor_frame(
        &[JAN_1, JAN_1, JAN_1, JAN_1, JAN_1, JAN_1],
        &["A", "B", "C", "D", "E", "F"],
        &[
            Some(10.0),
            Some(20.0),
            Some(30.0),
            Some(40.0),
            Some(50.0),
            Some(60.0),
        ],
    );

    let result =
        quantize_factor(&factor, QuantizeOptions::quantiles(3)).expect("quantiles should compute");

    let quantiles = result
        .column("factor_quantile")
        .unwrap()
        .u32()
        .unwrap()
        .iter()
        .collect::<Vec<_>>();
    assert!(quantiles.iter().all(|value| matches!(value, Some(1..=3))));
    assert!(quantiles.contains(&Some(1)));
    assert!(quantiles.contains(&Some(2)));
    assert!(quantiles.contains(&Some(3)));
}

#[test]
fn quantize_factor_property_row_order_invariance() {
    let ascending = factor_frame(
        &[JAN_1, JAN_1, JAN_1, JAN_1],
        &["A", "B", "C", "D"],
        &[Some(1.0), Some(2.0), Some(3.0), Some(4.0)],
    );
    let shuffled = factor_frame(
        &[JAN_1, JAN_1, JAN_1, JAN_1],
        &["C", "A", "D", "B"],
        &[Some(3.0), Some(1.0), Some(4.0), Some(2.0)],
    );

    let left =
        quantize_factor(&ascending, QuantizeOptions::quantiles(2)).expect("left should compute");
    let right =
        quantize_factor(&shuffled, QuantizeOptions::quantiles(2)).expect("right should compute");

    assert_eq!(
        asset_quantile_pairs(&left),
        vec![
            ("A".to_owned(), Some(1)),
            ("B".to_owned(), Some(1)),
            ("C".to_owned(), Some(2)),
            ("D".to_owned(), Some(2)),
        ]
    );
    assert_eq!(asset_quantile_pairs(&left), asset_quantile_pairs(&right));
}

#[test]
fn quantize_factor_assigns_deterministic_quantiles_by_date() {
    let factor = factor_frame(
        &[JAN_1, JAN_1, JAN_1, JAN_1],
        &["D", "B", "A", "C"],
        &[Some(4.0), Some(2.0), Some(1.0), Some(3.0)],
    );

    let result =
        quantize_factor(&factor, QuantizeOptions::quantiles(2)).expect("quantiles should compute");

    let assets: Vec<_> = result
        .column("asset")
        .unwrap()
        .str()
        .unwrap()
        .iter()
        .collect();
    let quantiles: Vec<_> = result
        .column("factor_quantile")
        .unwrap()
        .u32()
        .unwrap()
        .iter()
        .collect();
    assert_eq!(assets, vec![Some("A"), Some("B"), Some("C"), Some("D")]);
    assert_eq!(quantiles, vec![Some(1), Some(1), Some(2), Some(2)]);
}

#[test]
fn quantize_factor_can_partition_by_group() {
    let factor = grouped_factor_frame(
        &[JAN_1, JAN_1, JAN_1, JAN_1],
        &["A", "B", "C", "D"],
        &["g1", "g1", "g2", "g2"],
        &[Some(1.0), Some(2.0), Some(10.0), Some(20.0)],
    );

    let result = quantize_factor(&factor, QuantizeOptions::quantiles(2).with_by_group(true))
        .expect("group quantiles should compute");

    let quantiles: Vec<_> = result
        .column("factor_quantile")
        .unwrap()
        .u32()
        .unwrap()
        .iter()
        .collect();
    assert_eq!(quantiles, vec![Some(1), Some(2), Some(1), Some(2)]);
}

#[test]
fn quantize_factor_supports_explicit_bins() {
    let factor = factor_frame(
        &[JAN_1, JAN_1, JAN_1, JAN_1],
        &["A", "B", "C", "D"],
        &[Some(-2.0), Some(-0.5), Some(0.5), Some(2.0)],
    );

    let result = quantize_factor(&factor, QuantizeOptions::bins(vec![-3.0, 0.0, 3.0]))
        .expect("explicit bins should compute");

    let bins: Vec<_> = result
        .column("factor_quantile")
        .unwrap()
        .u32()
        .unwrap()
        .iter()
        .collect();
    assert_eq!(bins, vec![Some(1), Some(1), Some(2), Some(2)]);
}

#[test]
fn alphalens_quantize_factor_accepts_single_integer_bin_for_events() {
    let factor = factor_frame(
        &[JAN_1, JAN_1, JAN_1],
        &["A", "B", "C"],
        &[Some(1.0), Some(4.0), Some(7.0)],
    );

    let result = quantize_factor(&factor, QuantizeOptions::bin_count(1))
        .expect("single integer bin should assign every finite event to bin one");

    assert_eq!(
        result
            .column("factor_quantile")
            .unwrap()
            .u32()
            .unwrap()
            .iter()
            .collect::<Vec<_>>(),
        vec![Some(1), Some(1), Some(1)]
    );
}

#[test]
fn alphalens_quantize_factor_supports_quantile_edge_lists() {
    let factor = factor_frame(
        &[JAN_1, JAN_1, JAN_1, JAN_1, JAN_2, JAN_2, JAN_2, JAN_2],
        &["A", "B", "C", "D", "A", "B", "C", "D"],
        &[
            Some(1.0),
            Some(2.0),
            Some(3.0),
            Some(4.0),
            Some(4.0),
            Some(3.0),
            Some(2.0),
            Some(1.0),
        ],
    );

    let full = quantize_factor(
        &factor,
        QuantizeOptions::quantile_edges(vec![0.0, 0.25, 0.5, 0.75, 1.0]),
    )
    .unwrap();
    let asymmetric = quantize_factor(
        &factor,
        QuantizeOptions::quantile_edges(vec![0.0, 0.5, 0.75, 1.0]),
    )
    .unwrap();
    let interior = quantize_factor(
        &factor,
        QuantizeOptions::quantile_edges(vec![0.25, 0.5, 0.75]),
    )
    .unwrap();

    assert_eq!(
        full.column("factor_quantile")
            .unwrap()
            .u32()
            .unwrap()
            .iter()
            .collect::<Vec<_>>(),
        vec![
            Some(1),
            Some(2),
            Some(3),
            Some(4),
            Some(4),
            Some(3),
            Some(2),
            Some(1),
        ]
    );
    assert_eq!(
        asymmetric
            .column("factor_quantile")
            .unwrap()
            .u32()
            .unwrap()
            .iter()
            .collect::<Vec<_>>(),
        vec![
            Some(1),
            Some(1),
            Some(2),
            Some(3),
            Some(3),
            Some(2),
            Some(1),
            Some(1),
        ]
    );
    assert_eq!(
        interior
            .column("factor_quantile")
            .unwrap()
            .u32()
            .unwrap()
            .iter()
            .collect::<Vec<_>>(),
        vec![None, Some(1), Some(2), None, None, Some(2), Some(1), None,]
    );
}

#[test]
fn alphalens_quantize_factor_maps_sparse_event_partitions_to_extreme_quantiles() {
    let factor = factor_frame(
        &[JAN_1, JAN_1, JAN_2, JAN_2],
        &["A", "B", "A", "B"],
        &[Some(1.0), Some(6.0), Some(4.0), Some(7.0)],
    );

    let result = quantize_factor(&factor, QuantizeOptions::quantiles(4)).unwrap();

    assert_eq!(
        result
            .column("factor_quantile")
            .unwrap()
            .u32()
            .unwrap()
            .iter()
            .collect::<Vec<_>>(),
        vec![Some(1), Some(4), Some(1), Some(4)]
    );
}

#[test]
fn quantize_factor_explicit_bins_match_pandas_cut_right_closed_boundaries() {
    let factor = factor_frame(
        &[JAN_1, JAN_1, JAN_1],
        &["A", "B", "C"],
        &[Some(-3.0), Some(0.0), Some(3.0)],
    );

    let result = quantize_factor(&factor, QuantizeOptions::bins(vec![-3.0, 0.0, 3.0]))
        .expect("explicit bin boundaries should compute deterministically");

    let bins: Vec<_> = result
        .column("factor_quantile")
        .unwrap()
        .u32()
        .unwrap()
        .iter()
        .collect();
    assert_eq!(bins, vec![None, Some(1), Some(2)]);
}

#[test]
fn clean_factor_preserves_numeric_forward_return_column_order() {
    let factor = factor_frame(
        &[JAN_1, JAN_1, JAN_1, JAN_1, JAN_1],
        &["A", "B", "C", "D", "E"],
        &[Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0)],
    );
    let returns = DataFrame::new(
        5,
        vec![
            datetime_column(&[JAN_1, JAN_1, JAN_1, JAN_1, JAN_1]),
            Series::new("asset".into(), &["A", "B", "C", "D", "E"]).into(),
            Series::new("forward_return_10D".into(), &[0.10, 0.11, 0.12, 0.13, 0.14]).into(),
            Series::new("forward_return_1D".into(), &[0.01, 0.02, 0.03, 0.04, 0.05]).into(),
            Series::new("forward_return_5D".into(), &[0.05, 0.06, 0.07, 0.08, 0.09]).into(),
        ],
    )
    .expect("returns frame should be valid");

    let clean = ferric_alpha::get_clean_factor(
        &factor,
        &returns,
        QuantizeOptions::quantiles(5),
        CleanOptions { max_loss: 0.0 },
    )
    .expect("clean factor should compute");

    assert_eq!(
        clean.frame.get_column_names(),
        vec![
            "date",
            "asset",
            "factor",
            "factor_quantile",
            "forward_return_1D",
            "forward_return_5D",
            "forward_return_10D",
        ]
    );
}

#[test]
fn quantize_factor_uses_asset_name_to_stabilize_ties() {
    let unsorted = factor_frame(
        &[JAN_1, JAN_1, JAN_1, JAN_1],
        &["D", "B", "A", "C"],
        &[Some(1.0), Some(1.0), Some(1.0), Some(1.0)],
    );
    let sorted = factor_frame(
        &[JAN_1, JAN_1, JAN_1, JAN_1],
        &["A", "B", "C", "D"],
        &[Some(1.0), Some(1.0), Some(1.0), Some(1.0)],
    );

    let left = quantize_factor(&unsorted, QuantizeOptions::quantiles(2))
        .expect("unsorted input should compute");
    let right = quantize_factor(&sorted, QuantizeOptions::quantiles(2))
        .expect("sorted input should compute");

    let left_quantiles: Vec<_> = left
        .column("factor_quantile")
        .unwrap()
        .u32()
        .unwrap()
        .iter()
        .collect();
    let right_quantiles: Vec<_> = right
        .column("factor_quantile")
        .unwrap()
        .u32()
        .unwrap()
        .iter()
        .collect();
    assert_eq!(left_quantiles, right_quantiles);
}

#[test]
fn quantize_factor_zero_aware_partitions_positive_and_negative_values() {
    let factor = factor_frame(
        &[JAN_1, JAN_1, JAN_1, JAN_1],
        &["A", "B", "C", "D"],
        &[Some(-2.0), Some(-1.0), Some(1.0), Some(2.0)],
    );

    let result = quantize_factor(&factor, QuantizeOptions::quantiles(4).with_zero_aware(true))
        .expect("zero-aware quantiles should compute");

    let quantiles: Vec<_> = result
        .column("factor_quantile")
        .unwrap()
        .u32()
        .unwrap()
        .iter()
        .collect();
    assert_eq!(quantiles, vec![Some(1), Some(2), Some(3), Some(4)]);
}

#[test]
fn quantize_factor_zero_aware_treats_zero_as_non_negative() {
    let factor = factor_frame(
        &[JAN_1, JAN_1, JAN_1, JAN_1],
        &["A", "B", "C", "D"],
        &[Some(-2.0), Some(-1.0), Some(0.0), Some(1.0)],
    );

    let result = quantize_factor(&factor, QuantizeOptions::quantiles(4).with_zero_aware(true))
        .expect("zero-aware quantiles should compute");

    let quantiles: Vec<_> = result
        .column("factor_quantile")
        .unwrap()
        .u32()
        .unwrap()
        .iter()
        .collect();
    assert_eq!(quantiles, vec![Some(1), Some(2), Some(3), Some(4)]);
}

#[test]
fn quantize_factor_rejects_odd_zero_aware_quantiles() {
    let factor = factor_frame(&[JAN_1, JAN_1], &["A", "B"], &[Some(-1.0), Some(1.0)]);

    let error = quantize_factor(&factor, QuantizeOptions::quantiles(3).with_zero_aware(true))
        .expect_err("odd zero-aware quantiles should fail");

    assert!(matches!(error, FerricAlphaError::InvalidQuantiles { .. }));
}

#[test]
fn quantize_factor_marks_insufficient_partitions_null() {
    let factor = factor_frame(&[JAN_1], &["A"], &[Some(1.0)]);

    let result = quantize_factor(&factor, QuantizeOptions::quantiles(2))
        .expect("insufficient partition should be represented as quantile loss");

    assert_eq!(
        result
            .column("factor_quantile")
            .unwrap()
            .u32()
            .unwrap()
            .get(0),
        None
    );
}

#[test]
fn clean_factor_and_forward_returns_reports_return_and_quantile_loss() {
    let factor = factor_frame(
        &[JAN_1, JAN_1, JAN_2, JAN_2],
        &["A", "B", "A", "B"],
        &[Some(1.0), Some(2.0), Some(1.5), None],
    );
    let prices = price_frame(
        &[JAN_1, JAN_2, JAN_1, JAN_2],
        &["A", "A", "B", "B"],
        &[Some(100.0), Some(110.0), Some(50.0), Some(55.0)],
    );

    let clean = get_clean_factor_and_forward_returns(
        &factor,
        &prices,
        &[Period::new(1).unwrap()],
        QuantizeOptions::quantiles(2),
        CleanOptions { max_loss: 0.75 },
    )
    .expect("cleaning should allow configured loss");

    assert_eq!(clean.loss.input_rows, 4);
    assert_eq!(clean.loss.forward_return_loss, 2);
    assert_eq!(clean.loss.quantile_loss, 2);
    assert_eq!(clean.frame.height(), 2);
    assert!(clean.frame.column("factor_quantile").is_ok());
    assert!(clean.frame.column("forward_return_1D").is_ok());
}

#[test]
fn get_clean_factor_rejects_timezone_mismatch_with_precomputed_returns() {
    let factor = DataFrame::new(
        1,
        vec![
            utc_datetime_column(&[JAN_1]),
            Series::new("asset".into(), &["A"]).into(),
            Series::new("factor".into(), &[1.0]).into(),
        ],
    )
    .expect("timezone test factor should be valid");
    let returns = DataFrame::new(
        1,
        vec![
            datetime_column(&[JAN_1]),
            Series::new("asset".into(), &["A"]).into(),
            Series::new("forward_return_1D".into(), &[0.1]).into(),
        ],
    )
    .expect("returns frame should be valid");

    let error = ferric_alpha::get_clean_factor(
        &factor,
        &returns,
        QuantizeOptions::quantiles(2),
        CleanOptions { max_loss: 1.0 },
    )
    .expect_err("timezone mismatch should fail");

    assert!(matches!(error, FerricAlphaError::TimezoneMismatch));
}

#[test]
fn clean_factor_treats_nan_forward_returns_as_loss() {
    let factor = factor_frame(&[JAN_1], &["A"], &[Some(1.0)]);
    let returns = DataFrame::new(
        1,
        vec![
            datetime_column(&[JAN_1]),
            Series::new("asset".into(), &["A"]).into(),
            Series::new("forward_return_1D".into(), &[f64::NAN]).into(),
        ],
    )
    .expect("returns frame should be valid");

    let clean = ferric_alpha::get_clean_factor(
        &factor,
        &returns,
        QuantizeOptions::quantiles(2),
        CleanOptions { max_loss: 1.0 },
    )
    .expect("NaN returns should be counted as loss");

    assert_eq!(clean.loss.forward_return_loss, 1);
    assert_eq!(clean.frame.height(), 0);
}

#[test]
fn clean_factor_and_forward_returns_enforces_max_loss() {
    let factor = factor_frame(&[JAN_1, JAN_2], &["A", "A"], &[Some(1.0), Some(2.0)]);
    let prices = price_frame(&[JAN_1], &["A"], &[Some(100.0)]);

    let error = get_clean_factor_and_forward_returns(
        &factor,
        &prices,
        &[Period::new(1).unwrap()],
        QuantizeOptions::quantiles(2),
        CleanOptions { max_loss: 0.5 },
    )
    .expect_err("loss above threshold should fail");

    assert!(matches!(error, FerricAlphaError::MaxLossExceeded { .. }));
}

#[test]
fn clean_factor_rejects_invalid_max_loss() {
    let factor = factor_frame(&[JAN_1], &["A"], &[Some(1.0)]);
    let returns = DataFrame::new(
        1,
        vec![
            datetime_column(&[JAN_1]),
            Series::new("asset".into(), &["A"]).into(),
            Series::new("forward_return_1D".into(), &[0.1]).into(),
        ],
    )
    .expect("returns frame should be valid");

    let nan_error = ferric_alpha::get_clean_factor(
        &factor,
        &returns,
        QuantizeOptions::quantiles(2),
        CleanOptions { max_loss: f64::NAN },
    )
    .expect_err("NaN max_loss should fail");
    let above_one_error = ferric_alpha::get_clean_factor(
        &factor,
        &returns,
        QuantizeOptions::quantiles(2),
        CleanOptions { max_loss: 1.1 },
    )
    .expect_err("max_loss above 1 should fail");

    assert!(matches!(nan_error, FerricAlphaError::InvalidMaxLoss { .. }));
    assert!(matches!(
        above_one_error,
        FerricAlphaError::InvalidMaxLoss { .. }
    ));
}
