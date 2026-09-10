use polars::prelude::*;

pub const JAN_1: i64 = 1_704_067_200_000;
pub const JAN_2: i64 = 1_704_153_600_000;

pub fn datetime_column(values: &[i64]) -> Column {
    Series::new("date".into(), values)
        .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
        .expect("test dates should cast")
        .into()
}

#[allow(clippy::too_many_arguments)]
pub fn factor_data(
    dates: &[i64],
    assets: &[&str],
    groups: Option<&[&str]>,
    factors: &[Option<f64>],
    quantiles: &[Option<u32>],
    one_day: &[Option<f64>],
    five_day: &[Option<f64>],
    ten_day: &[Option<f64>],
) -> DataFrame {
    let mut columns = vec![
        datetime_column(dates),
        Series::new("asset".into(), assets).into(),
        Series::new("factor".into(), factors).into(),
    ];
    if let Some(groups) = groups {
        columns.push(Series::new("group".into(), groups).into());
    }
    columns.push(Series::new("factor_quantile".into(), quantiles).into());
    columns.push(Series::new("forward_return_1D".into(), one_day).into());
    columns.push(Series::new("forward_return_5D".into(), five_day).into());
    columns.push(Series::new("forward_return_10D".into(), ten_day).into());
    DataFrame::new(dates.len(), columns).expect("test factor data should build")
}

pub fn simple_factor_data() -> DataFrame {
    factor_data(
        &[JAN_1, JAN_1, JAN_1, JAN_2, JAN_2, JAN_2],
        &["A", "B", "C", "A", "B", "C"],
        Some(&["g1", "g1", "g2", "g1", "g1", "g2"]),
        &[
            Some(-1.0),
            Some(0.0),
            Some(1.0),
            Some(1.0),
            Some(0.0),
            Some(-1.0),
        ],
        &[Some(1), Some(3), Some(5), Some(5), Some(3), Some(1)],
        &[
            Some(-0.01),
            Some(0.0),
            Some(0.01),
            Some(-0.02),
            Some(0.0),
            Some(0.02),
        ],
        &[
            Some(-0.05),
            Some(0.0),
            Some(0.05),
            Some(-0.10),
            Some(0.0),
            Some(0.10),
        ],
        &[
            Some(-0.10),
            Some(0.0),
            Some(0.10),
            Some(-0.20),
            Some(0.0),
            Some(0.20),
        ],
    )
}
