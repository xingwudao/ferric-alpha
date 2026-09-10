use criterion::{Criterion, criterion_group, criterion_main};
use ferric_alpha::{
    EventReturnsOptions, FullTearSheetOptions, IcOptions, InformationTearSheetOptions,
    MeanReturnByQuantileOptions, Period, PortfolioOptions, ReturnOptions, ReturnsTearSheetOptions,
    SummaryTearSheetOptions, TearSheetData, TurnoverTearSheetOptions, WeightOptions,
    average_cumulative_return_by_quantile, create_full_tear_sheet_data,
    create_information_tear_sheet_data, create_returns_tear_sheet_data,
    create_summary_tear_sheet_data, create_turnover_tear_sheet_data, factor_cumulative_returns,
    factor_information_coefficient, factor_positions, factor_rank_autocorrelation, factor_returns,
    factor_weights, mean_return_by_quantile, quantile_turnover,
};
use polars::prelude::*;

fn benchmark_core_analytics(c: &mut Criterion) {
    let full = std::env::var_os("FERRIC_ALPHA_FULL_BENCH").is_some();
    for assets in [250, 1_000, 5_000] {
        for sessions in [252, 1_260] {
            for periods in [1, 3, 10] {
                for grouped in [false, true] {
                    if !full && (assets, sessions, periods) != (250, 252, 3) {
                        continue;
                    }
                    bench_case(c, assets, sessions, periods, grouped);
                }
            }
        }
    }
    report_bench_cases(c);
}

fn bench_case(c: &mut Criterion, assets: usize, sessions: usize, periods: usize, grouped: bool) {
    let label = if grouped { "grouped" } else { "ungrouped" };
    let frame = synthetic_frame(assets, sessions, periods, grouped);
    let prefix = format!("smoke/{assets}-assets/{sessions}-sessions/{periods}-periods/{label}");
    c.bench_function(&format!("{prefix}/ic"), |b| {
        b.iter(|| {
            factor_information_coefficient(&frame, IcOptions::default().with_by_group(grouped))
                .unwrap()
        });
    });
    c.bench_function(&format!("{prefix}/weights"), |b| {
        b.iter(|| factor_weights(&frame, WeightOptions::default()).unwrap());
    });
    c.bench_function(&format!("{prefix}/returns"), |b| {
        b.iter(|| factor_returns(&frame, ReturnOptions::default()).unwrap());
    });
    c.bench_function(&format!("{prefix}/quantiles"), |b| {
        b.iter(|| {
            mean_return_by_quantile(
                &frame,
                MeanReturnByQuantileOptions::default().with_by_date(true),
            )
            .unwrap()
        });
    });
    c.bench_function(&format!("{prefix}/turnover"), |b| {
        b.iter(|| quantile_turnover(&frame, 5, Period::new(1).unwrap()).unwrap());
    });
    c.bench_function(&format!("{prefix}/rank-autocorrelation"), |b| {
        b.iter(|| factor_rank_autocorrelation(&frame, Period::new(1).unwrap()).unwrap());
    });
    c.bench_function(&format!("{prefix}/factor-cumulative"), |b| {
        b.iter(|| {
            factor_cumulative_returns(&frame, Period::new(1).unwrap(), PortfolioOptions::default())
                .unwrap()
        });
    });
    c.bench_function(&format!("{prefix}/factor-positions"), |b| {
        b.iter(|| {
            factor_positions(
                &frame,
                Period::new(periods as u32).unwrap(),
                PortfolioOptions::default(),
                None,
            )
            .unwrap()
        });
    });
}

fn synthetic_frame(assets: usize, sessions: usize, periods: usize, grouped: bool) -> DataFrame {
    let mut dates = Vec::with_capacity(assets * sessions);
    let mut asset_names = Vec::with_capacity(assets * sessions);
    let mut groups = Vec::with_capacity(assets * sessions);
    let mut factors = Vec::with_capacity(assets * sessions);
    let mut quantiles = Vec::with_capacity(assets * sessions);
    let mut returns = vec![Vec::with_capacity(assets * sessions); periods];
    let start = 1_704_067_200_000_i64;

    for session in 0..sessions {
        let date = start + session as i64 * 86_400_000;
        for asset in 0..assets {
            dates.push(date);
            asset_names.push(format!("A{asset:05}"));
            groups.push(format!("G{}", asset % 5));
            let centered = asset as f64 - assets as f64 / 2.0;
            factors.push(Some(centered + (session % 7) as f64 * 0.01));
            quantiles.push(Some((asset % 5 + 1) as u32));
            for (period_index, output) in returns.iter_mut().enumerate() {
                let value = centered.signum() * 0.0001 * (period_index + 1) as f64
                    + (session % 11) as f64 * 0.00001;
                output.push(Some(value));
            }
        }
    }

    let mut columns = vec![
        Series::new("date".into(), dates)
            .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
            .unwrap()
            .into(),
        Series::new("asset".into(), asset_names).into(),
        Series::new("factor".into(), factors).into(),
    ];
    if grouped {
        columns.push(Series::new("group".into(), groups).into());
    }
    columns.push(Series::new("factor_quantile".into(), quantiles).into());
    for (index, values) in returns.into_iter().enumerate() {
        columns.push(Series::new(format!("forward_return_{}D", index + 1).into(), values).into());
    }
    DataFrame::new(assets * sessions, columns).unwrap()
}

fn report_bench_cases(c: &mut Criterion) {
    let frame = synthetic_frame(250, 252, 3, true);
    c.bench_function("report-smoke/summary", |b| {
        b.iter(|| {
            create_summary_tear_sheet_data(
                &frame,
                SummaryTearSheetOptions {
                    group_neutral: false,
                    turnover_periods: Some(vec![Period::new(1).unwrap(), Period::new(3).unwrap()]),
                    ..SummaryTearSheetOptions::default()
                },
            )
            .unwrap()
        });
    });
    c.bench_function("report-smoke/returns", |b| {
        b.iter(|| {
            create_returns_tear_sheet_data(
                &frame,
                ReturnsTearSheetOptions {
                    by_group: true,
                    ..ReturnsTearSheetOptions::default()
                },
            )
            .unwrap()
        });
    });
    c.bench_function("report-smoke/information", |b| {
        b.iter(|| {
            create_information_tear_sheet_data(
                &frame,
                InformationTearSheetOptions {
                    by_group: true,
                    rolling_window: 22,
                    ..InformationTearSheetOptions::default()
                },
            )
            .unwrap()
        });
    });
    c.bench_function("report-smoke/turnover", |b| {
        b.iter(|| {
            create_turnover_tear_sheet_data(
                &frame,
                TurnoverTearSheetOptions {
                    periods: Some(vec![Period::new(1).unwrap(), Period::new(3).unwrap()]),
                },
            )
            .unwrap()
        });
    });
    c.bench_function("report-smoke/full", |b| {
        b.iter(|| {
            create_full_tear_sheet_data(
                &frame,
                FullTearSheetOptions {
                    by_group: true,
                    turnover_periods: Some(vec![Period::new(1).unwrap(), Period::new(3).unwrap()]),
                    ic_rolling_window: 22,
                    ..FullTearSheetOptions::default()
                },
            )
            .unwrap()
        });
    });
    c.bench_function("report-smoke/full-json", |b| {
        b.iter(|| {
            let report = create_full_tear_sheet_data(
                &frame,
                FullTearSheetOptions {
                    by_group: true,
                    turnover_periods: Some(vec![Period::new(1).unwrap(), Period::new(3).unwrap()]),
                    ic_rolling_window: 22,
                    ..FullTearSheetOptions::default()
                },
            )
            .unwrap();
            TearSheetData::from(report).to_json().unwrap()
        });
    });

    let event_factor = synthetic_frame(50, 252, 1, true);
    let event_returns = synthetic_event_returns(50, 252);
    c.bench_function("report-smoke/event-average-cumulative", |b| {
        b.iter(|| {
            average_cumulative_return_by_quantile(
                &event_factor,
                &event_returns,
                EventReturnsOptions {
                    periods_before: 5,
                    periods_after: 15,
                    demeaned: true,
                    group_adjust: false,
                    by_group: false,
                },
            )
            .unwrap()
        });
    });
}

fn synthetic_event_returns(assets: usize, sessions: usize) -> DataFrame {
    let mut dates = Vec::with_capacity(assets * sessions);
    let mut asset_names = Vec::with_capacity(assets * sessions);
    let mut returns = Vec::with_capacity(assets * sessions);
    let start = 1_704_067_200_000_i64;

    for session in 0..sessions {
        let date = start + session as i64 * 86_400_000;
        for asset in 0..assets {
            dates.push(date);
            asset_names.push(format!("A{asset:05}"));
            returns.push(Some(
                (asset as f64 % 11.0 - 5.0) * 0.0002 + session as f64 * 0.000001,
            ));
        }
    }

    DataFrame::new(
        assets * sessions,
        vec![
            Series::new("date".into(), dates)
                .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
                .unwrap()
                .into(),
            Series::new("asset".into(), asset_names).into(),
            Series::new("return".into(), returns).into(),
        ],
    )
    .unwrap()
}

criterion_group!(benches, benchmark_core_analytics);
criterion_main!(benches);
