#[allow(dead_code)]
mod common;

use approx::assert_abs_diff_eq;
use ferric_alpha::{EventReturnsOptions, average_cumulative_return_by_quantile_from_prices};
use polars::prelude::*;

const DAY_MS: i64 = 86_400_000;
const JAN_15_2015: i64 = 1_421_280_000_000;
const JAN_18_2015: i64 = 1_421_539_200_000;
const JAN_21_2015: i64 = 1_421_798_400_000;

fn price_frame(start: i64, days: i64, assets: &[&str], bases: &[f64]) -> DataFrame {
    let mut dates = Vec::new();
    let mut tickers = Vec::new();
    let mut values = Vec::new();
    for i in 1..=days {
        let date = start + (i - 1) * DAY_MS;
        for (asset, base) in assets.iter().zip(bases) {
            dates.push(date);
            tickers.push(*asset);
            values.push(Some(base.powi(i as i32)));
        }
    }
    DataFrame::new(
        dates.len(),
        vec![
            common::datetime_column(&dates),
            Series::new("asset".into(), tickers).into(),
            Series::new("return".into(), values).into(),
        ],
    )
    .unwrap()
}

fn quantile_for(value: f64, quantiles: u32) -> u32 {
    if quantiles == 2 {
        if value <= 2.0 { 1 } else { 2 }
    } else {
        value as u32
    }
}

fn factor_frame(start: i64, rows: &[Vec<Option<f64>>], quantiles: u32) -> DataFrame {
    let assets = ["A", "B", "C", "D", "E", "F"];
    let mut dates = Vec::new();
    let mut tickers = Vec::new();
    let mut factors = Vec::new();
    let mut factor_quantiles = Vec::new();
    for (day, row) in rows.iter().enumerate() {
        for (asset, value) in assets.iter().zip(row) {
            let Some(value) = value else {
                continue;
            };
            dates.push(start + day as i64 * DAY_MS);
            tickers.push(*asset);
            factors.push(Some(*value));
            factor_quantiles.push(Some(quantile_for(*value, quantiles)));
        }
    }
    DataFrame::new(
        dates.len(),
        vec![
            common::datetime_column(&dates),
            Series::new("asset".into(), tickers).into(),
            Series::new("factor".into(), factors).into(),
            Series::new("factor_quantile".into(), factor_quantiles).into(),
        ],
    )
    .unwrap()
}

fn assert_quantile_matrix(
    frame: &DataFrame,
    quantiles: u32,
    before: u32,
    after: u32,
    expected_rows: &[Vec<f64>],
) {
    let qs = frame.column("factor_quantile").unwrap().u32().unwrap();
    let offsets = frame.column("offset").unwrap().i32().unwrap();
    let means = frame
        .column("mean_cumulative_return")
        .unwrap()
        .f64()
        .unwrap();
    let stds = frame
        .column("std_cumulative_return")
        .unwrap()
        .f64()
        .unwrap();
    let offset_values = (-(before as i32)..=after as i32).collect::<Vec<_>>();

    assert_eq!(expected_rows.len(), quantiles as usize * 2);
    for q in 1..=quantiles {
        let mean_expected = &expected_rows[(q as usize - 1) * 2];
        let std_expected = &expected_rows[(q as usize - 1) * 2 + 1];
        for (offset_index, offset) in offset_values.iter().enumerate() {
            let row = (0..frame.height())
                .find(|index| qs.get(*index) == Some(q) && offsets.get(*index) == Some(*offset))
                .expect("quantile offset row should exist");
            assert_abs_diff_eq!(
                means.get(row).unwrap(),
                mean_expected[offset_index],
                epsilon = 1e-5
            );
            assert_abs_diff_eq!(
                stds.get(row).unwrap_or(0.0),
                std_expected[offset_index],
                epsilon = 1e-5
            );
        }
    }
}

#[test]
fn alphalens_average_cumulative_return_by_quantile_fixture_matches_public_tests() {
    let cases = vec![
        (
            1,
            2,
            false,
            4,
            vec![
                vec![0.00512695, 0.00256348, 0.00128174, 6.40869e-4],
                vec![0.00579185, 0.00289592, 0.00144796, 7.23981e-4],
                vec![1.0, 1.0, 1.0, 1.0],
                vec![0.0, 0.0, 0.0, 0.0],
                vec![7.15814531, 8.94768164, 11.1846020, 13.9807526],
                vec![2.93784787, 3.67230984, 4.59038730, 5.73798413],
                vec![39.4519043, 59.1778564, 88.7667847, 133.150177],
                vec![28.3717330, 42.5575995, 63.8363992, 95.7545989],
            ],
        ),
        (
            1,
            2,
            true,
            4,
            vec![
                vec![-11.898667, -17.279462, -25.236885, -37.032252],
                vec![7.82587034, 11.5529583, 17.0996881, 25.3636472],
                vec![-10.903794, -16.282025, -24.238167, -36.032893],
                vec![7.82140124, 11.5507268, 17.0985737, 25.3630906],
                vec![-4.7456488, -8.3343438, -14.053565, -23.052140],
                vec![4.91184665, 7.91180853, 12.5481552, 19.6734224],
                vec![27.5481102, 41.8958311, 63.5286176, 96.1172844],
                vec![20.5510133, 31.0075980, 46.7385910, 70.3923129],
            ],
        ),
        (
            3,
            0,
            false,
            4,
            vec![
                vec![0.0205078125, 0.01025390625, 0.005126953125, 0.0025634765625],
                vec![
                    0.02316703549489275,
                    0.01158351774744638,
                    0.00579175887372319,
                    0.00289587943686159,
                ],
                vec![1.0, 1.0, 1.0, 1.0],
                vec![0.0, 0.0, 0.0, 0.0],
                vec![4.581213, 5.72651625, 7.15814531, 8.94768164],
                vec![1.88022296, 2.35027870, 2.93784870, 3.67230998],
                vec![17.534180, 26.301270, 39.4519043, 59.1778564],
                vec![12.609659, 18.914489, 28.3717330, 42.5575995],
            ],
        ),
        (
            0,
            3,
            true,
            4,
            vec![
                vec![-17.279462, -25.236885, -37.032252, -54.550061],
                vec![11.5529583, 17.0996881, 25.3636472, 37.6887906],
                vec![-16.282025, -24.238167, -36.032893, -53.550382],
                vec![11.5507268, 17.0985737, 25.3630906, 37.6885125],
                vec![-8.3343438, -14.053565, -23.052140, -37.074441],
                vec![7.91180853, 12.5481552, 19.6734224, 30.5748605],
                vec![41.8958311, 63.5286176, 96.1172844, 145.174884],
                vec![31.0075980, 46.7385910, 70.3923129, 105.944230],
            ],
        ),
        (
            3,
            3,
            false,
            2,
            vec![
                vec![
                    0.5102539, 0.50512695, 0.50256348, 0.50128174, 0.50064087, 0.50032043,
                    0.50016022,
                ],
                vec![
                    0.0115837, 0.00579185, 0.00289592, 1.44796e-3, 7.23981e-4, 3.61990e-4,
                    1.80995e-4,
                ],
                vec![
                    11.057696, 16.0138929, 23.3050248, 34.0627690, 49.9756934, 73.5654648,
                    108.600603,
                ],
                vec![
                    7.2389454, 10.6247239, 15.6450367, 23.1025693, 34.1977045, 50.7264595,
                    75.3771641,
                ],
            ],
        ),
        (
            3,
            3,
            true,
            2,
            vec![
                vec![
                    -5.273721, -7.754383, -11.40123, -16.78074, -24.73753, -36.53257, -54.05022,
                ],
                vec![
                    3.6239580, 5.3146000, 7.8236356, 11.551843, 17.099131, 25.363369, 37.688652,
                ],
                vec![
                    5.2737212, 7.7543830, 11.401231, 16.780744, 24.737526, 36.532572, 54.050221,
                ],
                vec![
                    3.6239580, 5.3146000, 7.8236356, 11.551843, 17.099131, 25.363369, 37.688652,
                ],
            ],
        ),
    ];
    let prices = price_frame(
        JAN_15_2015,
        18,
        &["A", "B", "C", "D"],
        &[1.25, 1.50, 1.00, 0.50],
    );
    let factor_rows = vec![
        vec![Some(3.0), Some(4.0), Some(2.0), Some(1.0)],
        vec![Some(3.0), Some(4.0), Some(2.0), Some(1.0)],
        vec![Some(3.0), Some(4.0), Some(2.0), Some(1.0)],
        vec![Some(3.0), Some(4.0), Some(2.0), Some(1.0)],
        vec![Some(3.0), Some(4.0), Some(2.0), Some(1.0)],
        vec![Some(3.0), Some(4.0), Some(2.0), Some(1.0)],
    ];

    for (before, after, demeaned, quantiles, expected) in cases {
        let factor = factor_frame(JAN_21_2015, &factor_rows, quantiles);
        let result = average_cumulative_return_by_quantile_from_prices(
            &factor,
            &prices,
            EventReturnsOptions {
                periods_before: before,
                periods_after: after,
                demeaned,
                group_adjust: false,
                by_group: false,
            },
        )
        .unwrap();
        assert_quantile_matrix(&result, quantiles, before, after, &expected);
    }
}

#[test]
fn alphalens_average_cumulative_return_by_quantile_varying_universe_fixture_matches_public_tests() {
    let cases = vec![
        (
            0,
            2,
            false,
            4,
            vec![
                vec![0.0292969, 0.0146484, 7.32422e-3],
                vec![0.0241851, 0.0120926, 6.04628e-3],
                vec![1.0, 1.0, 1.0],
                vec![0.0, 0.0, 0.0],
                vec![3.5190582, 4.3988228, 5.49852848],
                vec![1.0046375, 1.2557969, 1.56974616],
                vec![10.283203, 15.424805, 23.1372070],
                vec![5.2278892, 7.8418338, 11.7627508],
            ],
        ),
        (
            0,
            3,
            true,
            4,
            vec![
                vec![-3.6785927, -5.1949205, -7.4034407, -10.641996],
                vec![1.57386873, 2.28176590, 3.33616491, 4.90228915],
                vec![-2.7078896, -4.2095690, -6.4107649, -9.6456583],
                vec![1.55205002, 2.27087143, 3.33072273, 4.89956999],
                vec![-0.1888313, -0.8107462, -1.9122365, -3.7724977],
                vec![0.55371389, 1.02143924, 1.76795263, 2.94536298],
                vec![6.57531357, 10.2152357, 15.7264421, 24.0601522],
                vec![3.67596914, 5.57112656, 8.43221341, 12.7447568],
            ],
        ),
        (
            0,
            3,
            false,
            2,
            vec![
                vec![0.51464844, 0.50732422, 0.50366211, 0.50183105],
                vec![0.01209256, 0.00604628, 0.00302314, 0.00151157],
                vec![6.90113068, 9.91181374, 14.3178678, 20.7894856],
                vec![3.11499629, 4.54718783, 6.66416616, 9.80049950],
            ],
        ),
        (
            0,
            3,
            true,
            2,
            vec![
                vec![-3.1932411, -4.7022448, -6.9071028, -10.143827],
                vec![1.56295067, 2.27631715, 3.33344356, 4.90092953],
                vec![3.19324112, 4.70224476, 6.90710282, 10.1438273],
                vec![1.56295067, 2.27631715, 3.33344356, 4.90092953],
            ],
        ),
    ];
    let prices = price_frame(
        JAN_15_2015,
        11,
        &["A", "B", "C", "D", "E", "F"],
        &[1.25, 1.50, 1.00, 0.50, 1.50, 1.00],
    );
    let factor_rows = vec![
        vec![Some(3.0), Some(4.0), Some(2.0), Some(1.0), None, None],
        vec![Some(3.0), Some(4.0), Some(2.0), Some(1.0), None, None],
        vec![Some(3.0), None, None, Some(1.0), Some(4.0), Some(2.0)],
        vec![Some(3.0), None, None, Some(1.0), Some(4.0), Some(2.0)],
    ];

    for (before, after, demeaned, quantiles, expected) in cases {
        let factor = factor_frame(JAN_18_2015, &factor_rows, quantiles);
        let result = average_cumulative_return_by_quantile_from_prices(
            &factor,
            &prices,
            EventReturnsOptions {
                periods_before: before,
                periods_after: after,
                demeaned,
                group_adjust: false,
                by_group: false,
            },
        )
        .unwrap();
        assert_quantile_matrix(&result, quantiles, before, after, &expected);
    }
}
