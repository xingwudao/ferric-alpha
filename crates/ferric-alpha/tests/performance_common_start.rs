#[allow(dead_code)]
mod common;

use approx::assert_abs_diff_eq;
use ferric_alpha::{CommonStartReturnsOptions, common_start_returns};
use polars::prelude::*;

const DAY_MS: i64 = 86_400_000;
const JAN_17_2015: i64 = 1_421_452_800_000;
const JAN_21_2015: i64 = 1_421_798_400_000;

type CommonStartExpectedRow = (i32, f64, f64);
type CommonStartCase = (u32, u32, bool, bool, Vec<CommonStartExpectedRow>);

fn alphalens_common_start_factor() -> DataFrame {
    let mut dates = Vec::new();
    let mut assets = Vec::new();
    let mut factors = Vec::new();
    for day in 0..9 {
        for (asset, factor) in [("A", 3.0), ("B", 4.0), ("C", 2.0), ("D", 1.0)] {
            dates.push(JAN_21_2015 + day * DAY_MS);
            assets.push(asset);
            factors.push(Some(factor));
        }
    }
    DataFrame::new(
        dates.len(),
        vec![
            common::datetime_column(&dates),
            Series::new("asset".into(), assets).into(),
            Series::new("factor".into(), factors).into(),
        ],
    )
    .unwrap()
}

fn alphalens_common_start_prices() -> DataFrame {
    let mut dates = Vec::new();
    let mut assets = Vec::new();
    let mut returns = Vec::new();
    for i in 1..=17 {
        let date = JAN_17_2015 + (i - 1) * DAY_MS;
        for (asset, base) in [("A", 1.20_f64), ("B", 1.40), ("C", 0.90), ("D", 0.80)] {
            dates.push(date);
            assets.push(asset);
            returns.push(Some(base.powi(i as i32)));
        }
    }
    DataFrame::new(
        dates.len(),
        vec![
            common::datetime_column(&dates),
            Series::new("asset".into(), assets).into(),
            Series::new("return".into(), returns).into(),
        ],
    )
    .unwrap()
}

#[test]
fn alphalens_common_start_returns_fixture_matches_public_tests() {
    let factor = alphalens_common_start_factor();
    let prices = alphalens_common_start_prices();
    let cases: Vec<CommonStartCase> = vec![
        (
            2,
            3,
            false,
            false,
            vec![
                (-2, 4.93048307, 8.68843922),
                (-1, 6.60404312, 12.22369139),
                (0, 8.92068367, 17.1794088),
                (1, 12.1275523, 24.12861778),
                (2, 16.5694159, 33.8740100),
                (3, 22.7273233, 47.53995233),
            ],
        ),
        (
            3,
            2,
            false,
            true,
            vec![
                (-3, 0.0, 5.63219176),
                (-2, 0.0, 7.96515233),
                (-1, 0.0, 11.2420646),
                (0, 0.0, 15.8458720),
                (1, 0.0, 22.3134160),
                (2, 0.0, 31.3970961),
            ],
        ),
        (
            3,
            5,
            true,
            false,
            vec![
                (-3, 3.7228318, 2.6210478),
                (-2, 4.9304831, 3.6296796),
                (-1, 6.6040431, 5.0193734),
                (0, 8.9206837, 6.9404046),
                (1, 12.127552, 9.6023405),
                (2, 16.569416, 13.297652),
                (3, 22.727323, 18.434747),
                (4, 31.272682, 25.584180),
                (5, 34.358565, 25.497254),
            ],
        ),
        (
            1,
            4,
            true,
            true,
            vec![
                (-1, 0.0, 0.0),
                (0, 0.0, 0.0),
                (1, 0.0, 0.0),
                (2, 0.0, 0.0),
                (3, 0.0, 0.0),
                (4, 0.0, 0.0),
            ],
        ),
        (
            6,
            6,
            false,
            false,
            vec![
                (-6, 2.02679565, 2.38468223),
                (-5, 2.38769454, 3.22602748),
                (-4, 2.85413029, 4.36044469),
                (-3, 3.72283181, 6.16462715),
                (-2, 4.93048307, 8.68843922),
                (-1, 6.60404312, 12.2236914),
                (0, 8.92068367, 17.1794088),
                (1, 12.1275523, 24.1286178),
                (2, 16.5694159, 33.8740100),
                (3, 22.7273233, 47.5399523),
                (4, 31.2726821, 66.7013483),
                (5, 34.3585654, 70.1828776),
                (6, 37.9964585, 74.3294620),
            ],
        ),
        (
            6,
            6,
            false,
            true,
            vec![
                (-6, 0.0, 2.20770299),
                (-5, 0.0, 2.95942924),
                (-4, 0.0, 3.97022414),
                (-3, 0.0, 5.63219176),
                (-2, 0.0, 7.96515233),
                (-1, 0.0, 11.2420646),
                (0, 0.0, 15.8458720),
                (1, 0.0, 22.3134160),
                (2, 0.0, 31.3970962),
                (3, 0.0, 44.1512888),
                (4, 0.0, 62.0533954),
                (5, 0.0, 65.8668371),
                (6, 0.0, 70.4306483),
            ],
        ),
        (
            6,
            6,
            true,
            false,
            vec![
                (-6, 2.0267957, 0.9562173),
                (-5, 2.3876945, 1.3511898),
                (-4, 2.8541303, 1.8856194),
                (-3, 3.7228318, 2.6210478),
                (-2, 4.9304831, 3.6296796),
                (-1, 6.6040431, 5.0193734),
                (0, 8.9206837, 6.9404046),
                (1, 12.127552, 9.6023405),
                (2, 16.569416, 13.297652),
                (3, 22.727323, 18.434747),
                (4, 31.272682, 25.584180),
                (5, 34.358565, 25.497254),
                (6, 37.996459, 25.198051),
            ],
        ),
        (
            6,
            6,
            true,
            true,
            vec![
                (-6, 0.0, 0.0),
                (-5, 0.0, 0.0),
                (-4, 0.0, 0.0),
                (-3, 0.0, 0.0),
                (-2, 0.0, 0.0),
                (-1, 0.0, 0.0),
                (0, 0.0, 0.0),
                (1, 0.0, 0.0),
                (2, 0.0, 0.0),
                (3, 0.0, 0.0),
                (4, 0.0, 0.0),
                (5, 0.0, 0.0),
                (6, 0.0, 0.0),
            ],
        ),
    ];

    for (before, after, mean_by_date, demeaned, expected) in cases {
        let result = common_start_returns(
            &factor,
            &prices,
            CommonStartReturnsOptions::default()
                .with_periods_before(before)
                .with_periods_after(after)
                .with_cumulative(true)
                .with_mean_by_date(mean_by_date)
                .with_demeaned(demeaned),
        )
        .unwrap();

        let offsets = result.column("offset").unwrap().i32().unwrap();
        let means = result.column("mean").unwrap().f64().unwrap();
        let stds = result.column("std").unwrap().f64().unwrap();
        assert_eq!(result.height(), expected.len());
        for (index, (offset, mean, std)) in expected.into_iter().enumerate() {
            assert_eq!(offsets.get(index), Some(offset));
            assert_abs_diff_eq!(means.get(index).unwrap(), mean, epsilon = 1e-6);
            assert_abs_diff_eq!(stds.get(index).unwrap(), std, epsilon = 1e-6);
        }
    }
}
