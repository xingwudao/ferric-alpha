use std::collections::{BTreeMap, VecDeque};

use polars::prelude::{DataFrame, DataType};
use serde::{Deserialize, Serialize};

use crate::performance::{IcOptions, MeanIcOptions, TimeBucket};
use crate::report::{
    ColumnRole, DisplayFormat, DisplayUnit, ReportColumn, ReportColumnData, ReportKind,
    ReportMetadata, ReportTable, ReportTimeUnit,
};
use crate::{
    FerricAlphaError, Result, biased_excess_kurtosis, biased_skew, factor_information_coefficient,
    mean_information_coefficient, normal_qq, one_sample_t_test, sample_std,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct InformationTearSheetOptions {
    pub group_neutral: bool,
    pub by_group: bool,
    pub rolling_window: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InformationReportData {
    pub summary: ReportTable,
    pub ic_by_date: ReportTable,
    pub ic_rolling: ReportTable,
    pub ic_monthly: ReportTable,
    pub ic_by_group: Option<ReportTable>,
    pub qq_normal: ReportTable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InformationTearSheetData {
    pub metadata: ReportMetadata,
    pub options: InformationTearSheetOptions,
    pub data: InformationReportData,
}

#[derive(Debug, Clone)]
struct IcRow {
    date: i64,
    period: String,
    value: Option<f64>,
}

impl Default for InformationTearSheetOptions {
    fn default() -> Self {
        Self {
            group_neutral: false,
            by_group: false,
            rolling_window: 22,
        }
    }
}

pub fn create_information_tear_sheet_data(
    factor_data: &DataFrame,
    options: InformationTearSheetOptions,
) -> Result<InformationTearSheetData> {
    if options.rolling_window == 0 {
        return Err(FerricAlphaError::InvalidOption {
            option: "rolling_window",
            reason: "must be positive",
        });
    }

    let metadata = ReportMetadata::from_factor_data(factor_data, ReportKind::Information)?;
    let date_time_unit = metadata.date_time_unit();
    let date_timezone = metadata.date_timezone().map(ToOwned::to_owned);
    let ic = factor_information_coefficient(
        factor_data,
        IcOptions::default().with_group_adjust(options.group_neutral),
    )?;
    let rows = collect_ic_rows(&ic)?;

    let summary = summary_table(&rows)?;
    let ic_by_date = ic_by_date_table(&rows, date_time_unit, date_timezone.clone())?;
    let ic_rolling = rolling_table(
        &rows,
        options.rolling_window,
        date_time_unit,
        date_timezone.clone(),
    )?;
    let monthly = mean_information_coefficient(
        factor_data,
        MeanIcOptions::default()
            .with_group_adjust(options.group_neutral)
            .with_by_time(Some(TimeBucket::Monthly)),
    )?;
    let ic_monthly = mean_ic_table(
        &monthly,
        "information.ic_monthly",
        "Monthly Mean Information Coefficient",
        MeanIcShape::Monthly,
    )?;
    let ic_by_group = options
        .by_group
        .then(|| {
            mean_information_coefficient(
                factor_data,
                MeanIcOptions::default()
                    .with_group_adjust(options.group_neutral)
                    .with_by_group(true),
            )
            .and_then(|frame| {
                mean_ic_table(
                    &frame,
                    "information.ic_by_group",
                    "Mean Information Coefficient by Group",
                    MeanIcShape::Group,
                )
            })
        })
        .transpose()?;
    let qq_normal = qq_table(&rows)?;

    Ok(InformationTearSheetData {
        metadata,
        options,
        data: InformationReportData {
            summary,
            ic_by_date,
            ic_rolling,
            ic_monthly,
            ic_by_group,
            qq_normal,
        },
    })
}

fn collect_ic_rows(frame: &DataFrame) -> Result<Vec<IcRow>> {
    let dates = frame.column("date")?.as_materialized_series().datetime()?;
    let periods = frame.column("period")?.as_materialized_series().str()?;
    let values = frame.column("ic")?.as_materialized_series().f64()?;
    let mut rows = Vec::with_capacity(frame.height());
    for index in 0..frame.height() {
        rows.push(IcRow {
            date: dates
                .physical()
                .get(index)
                .expect("generated IC date should not be null"),
            period: periods
                .get(index)
                .expect("generated IC period should not be null")
                .to_owned(),
            value: values.get(index).filter(|value| value.is_finite()),
        });
    }
    rows.sort_by(|left, right| {
        left.date
            .cmp(&right.date)
            .then(left.period.cmp(&right.period))
    });
    Ok(rows)
}

fn summary_table(rows: &[IcRow]) -> Result<ReportTable> {
    let mut grouped = BTreeMap::<String, Vec<f64>>::new();
    for row in rows {
        let values = grouped.entry(row.period.clone()).or_default();
        if let Some(value) = row.value {
            values.push(value);
        }
    }

    let mut out = Vec::with_capacity(grouped.len());

    for (period, values) in grouped {
        let count = values.len();
        let mean = (!values.is_empty()).then(|| values.iter().sum::<f64>() / count as f64);
        let std = sample_std(&values)?;
        let t_test = one_sample_t_test(&values, 0.0)?;
        out.push((
            period,
            mean,
            std,
            match (mean, std) {
                (Some(mean), Some(std)) if std != 0.0 => Some(mean / std),
                _ => None,
            },
            t_test.statistic,
            t_test.p_value,
            biased_skew(&values)?,
            biased_excess_kurtosis(&values)?,
            Some(count as u64),
        ));
    }
    out.sort_by(|left, right| left.0.cmp(&right.0));

    ReportTable::new(
        "information.summary",
        "Information Analysis",
        vec!["period".to_string()],
        vec![
            string_column(
                "period",
                "Period",
                ColumnRole::Dimension,
                out.iter().map(|row| Some(row.0.clone())).collect(),
            ),
            float_column(
                "ic_mean",
                "IC Mean",
                ColumnRole::Statistic,
                out.iter().map(|row| row.1).collect(),
                6,
            ),
            float_column(
                "ic_std",
                "IC Std",
                ColumnRole::Statistic,
                out.iter().map(|row| row.2).collect(),
                6,
            ),
            float_column(
                "risk_adjusted_ic",
                "Risk-Adjusted IC",
                ColumnRole::Statistic,
                out.iter().map(|row| row.3).collect(),
                6,
            ),
            float_column(
                "t_stat",
                "T-Statistic",
                ColumnRole::Statistic,
                out.iter().map(|row| row.4).collect(),
                6,
            ),
            float_column(
                "p_value",
                "P-Value",
                ColumnRole::Statistic,
                out.iter().map(|row| row.5).collect(),
                6,
            ),
            float_column(
                "skew",
                "Skew",
                ColumnRole::Statistic,
                out.iter().map(|row| row.6).collect(),
                6,
            ),
            float_column(
                "excess_kurtosis",
                "Excess Kurtosis",
                ColumnRole::Statistic,
                out.iter().map(|row| row.7).collect(),
                6,
            ),
            count_column("count", "Count", out.iter().map(|row| row.8).collect()),
        ],
    )
}

fn ic_by_date_table(
    rows: &[IcRow],
    time_unit: ReportTimeUnit,
    timezone: Option<String>,
) -> Result<ReportTable> {
    ReportTable::new(
        "information.ic_by_date",
        "Information Coefficient",
        vec!["date".to_string(), "period".to_string()],
        vec![
            datetime_column(
                "date",
                "Date",
                rows.iter().map(|row| Some(row.date)).collect(),
                time_unit,
                timezone,
            ),
            string_column(
                "period",
                "Period",
                ColumnRole::Dimension,
                rows.iter().map(|row| Some(row.period.clone())).collect(),
            ),
            float_column(
                "ic",
                "Information Coefficient",
                ColumnRole::Measure,
                rows.iter().map(|row| row.value).collect(),
                6,
            ),
        ],
    )
}

fn rolling_table(
    rows: &[IcRow],
    window: u32,
    time_unit: ReportTimeUnit,
    timezone: Option<String>,
) -> Result<ReportTable> {
    let mut by_period = BTreeMap::<String, Vec<&IcRow>>::new();
    for row in rows {
        by_period.entry(row.period.clone()).or_default().push(row);
    }

    let mut out = Vec::<(i64, String, Option<f64>, Option<u64>)>::new();
    let window = window as usize;
    for (period, mut period_rows) in by_period {
        period_rows.sort_by_key(|row| row.date);
        let mut window_values = VecDeque::<Option<f64>>::new();
        let mut finite_sum = 0.0;
        let mut finite_count = 0_usize;

        for row in period_rows {
            if let Some(value) = row.value {
                finite_sum += value;
                finite_count += 1;
            }
            window_values.push_back(row.value);
            if window_values.len() > window
                && let Some(removed) = window_values.pop_front().flatten()
            {
                finite_sum -= removed;
                finite_count -= 1;
            }
            let rolling = (window_values.len() == window && finite_count == window)
                .then(|| finite_sum / window as f64);
            out.push((row.date, period.clone(), rolling, Some(finite_count as u64)));
        }
    }
    out.sort_by(|left, right| left.0.cmp(&right.0).then(left.1.cmp(&right.1)));

    ReportTable::new(
        "information.ic_rolling",
        "Rolling Mean Information Coefficient",
        vec!["date".to_string(), "period".to_string()],
        vec![
            datetime_column(
                "date",
                "Date",
                out.iter().map(|row| Some(row.0)).collect(),
                time_unit,
                timezone,
            ),
            string_column(
                "period",
                "Period",
                ColumnRole::Dimension,
                out.iter().map(|row| Some(row.1.clone())).collect(),
            ),
            float_column(
                "rolling_mean_ic",
                "Rolling Mean IC",
                ColumnRole::Measure,
                out.iter().map(|row| row.2).collect(),
                6,
            ),
            count_column("count", "Count", out.iter().map(|row| row.3).collect()),
        ],
    )
}

#[derive(Debug, Clone, Copy)]
enum MeanIcShape {
    Monthly,
    Group,
}

fn mean_ic_table(
    frame: &DataFrame,
    id: &'static str,
    title: &'static str,
    shape: MeanIcShape,
) -> Result<ReportTable> {
    let periods = frame.column("period")?.as_materialized_series().str()?;
    let means = frame.column("mean_ic")?.as_materialized_series().f64()?;
    let counts = frame.column("count")?.as_materialized_series().u32()?;

    match shape {
        MeanIcShape::Monthly => {
            let bucket_values = month_bucket_values(frame)?;
            let mut out = (0..frame.height())
                .map(|index| {
                    (
                        bucket_values[index],
                        periods.get(index).map(ToOwned::to_owned),
                        means.get(index),
                        counts.get(index).map(u64::from),
                    )
                })
                .collect::<Vec<_>>();
            out.sort_by(|left, right| left.0.cmp(&right.0).then(left.1.cmp(&right.1)));
            ReportTable::new(
                id,
                title,
                vec!["time_bucket".to_string(), "period".to_string()],
                vec![
                    ReportColumn::new(
                        "time_bucket",
                        "Time Bucket",
                        ColumnRole::Dimension,
                        None,
                        ReportColumnData::Int32(out.iter().map(|row| row.0).collect()),
                    ),
                    string_column(
                        "period",
                        "Period",
                        ColumnRole::Dimension,
                        out.iter().map(|row| row.1.clone()).collect(),
                    ),
                    float_column(
                        "mean_ic",
                        "Mean IC",
                        ColumnRole::Measure,
                        out.iter().map(|row| row.2).collect(),
                        6,
                    ),
                    count_column("count", "Count", out.iter().map(|row| row.3).collect()),
                ],
            )
        }
        MeanIcShape::Group => {
            let groups = frame.column("group")?.as_materialized_series().str()?;
            let mut out = (0..frame.height())
                .map(|index| {
                    (
                        groups.get(index).map(ToOwned::to_owned),
                        periods.get(index).map(ToOwned::to_owned),
                        means.get(index),
                        counts.get(index).map(u64::from),
                    )
                })
                .collect::<Vec<_>>();
            out.sort_by(|left, right| left.0.cmp(&right.0).then(left.1.cmp(&right.1)));
            ReportTable::new(
                id,
                title,
                vec!["group".to_string(), "period".to_string()],
                vec![
                    string_column(
                        "group",
                        "Group",
                        ColumnRole::Dimension,
                        out.iter().map(|row| row.0.clone()).collect(),
                    ),
                    string_column(
                        "period",
                        "Period",
                        ColumnRole::Dimension,
                        out.iter().map(|row| row.1.clone()).collect(),
                    ),
                    float_column(
                        "mean_ic",
                        "Mean IC",
                        ColumnRole::Measure,
                        out.iter().map(|row| row.2).collect(),
                        6,
                    ),
                    count_column("count", "Count", out.iter().map(|row| row.3).collect()),
                ],
            )
        }
    }
}

fn month_bucket_values(frame: &DataFrame) -> Result<Vec<Option<i32>>> {
    let series = frame.column("time_bucket")?.as_materialized_series();
    match series.dtype() {
        DataType::Date => {
            let buckets = series.date()?;
            Ok((0..frame.height())
                .map(|index| buckets.phys.get(index))
                .collect())
        }
        DataType::Int32 => {
            let buckets = series.i32()?;
            Ok((0..frame.height())
                .map(|index| buckets.get(index))
                .collect())
        }
        actual => Err(FerricAlphaError::InvalidDtype {
            column: "time_bucket",
            expected: "Date or Int32",
            actual: actual.to_string(),
        }),
    }
}

fn qq_table(rows: &[IcRow]) -> Result<ReportTable> {
    let mut grouped = BTreeMap::<String, Vec<f64>>::new();
    for row in rows {
        if let Some(value) = row.value {
            grouped.entry(row.period.clone()).or_default().push(value);
        }
    }

    let mut out = Vec::<(String, f64, f64, f64)>::new();
    for (period, values) in grouped {
        for point in normal_qq(&values)? {
            out.push((
                period.clone(),
                point.probability,
                point.theoretical,
                point.observed,
            ));
        }
    }
    out.sort_by(|left, right| {
        left.0
            .cmp(&right.0)
            .then(left.1.total_cmp(&right.1))
            .then(left.2.total_cmp(&right.2))
    });

    ReportTable::new(
        "information.qq_normal",
        "Normal Q-Q Data",
        vec!["period".to_string(), "probability".to_string()],
        vec![
            string_column(
                "period",
                "Period",
                ColumnRole::Dimension,
                out.iter().map(|row| Some(row.0.clone())).collect(),
            ),
            float_column(
                "probability",
                "Probability",
                ColumnRole::Dimension,
                out.iter().map(|row| Some(row.1)).collect(),
                6,
            ),
            float_column(
                "theoretical",
                "Theoretical Quantile",
                ColumnRole::Measure,
                out.iter().map(|row| Some(row.2)).collect(),
                6,
            ),
            float_column(
                "observed",
                "Observed Quantile",
                ColumnRole::Measure,
                out.iter().map(|row| Some(row.3)).collect(),
                6,
            ),
        ],
    )
}

fn string_column(
    name: &'static str,
    label: &'static str,
    role: ColumnRole,
    values: Vec<Option<String>>,
) -> ReportColumn {
    ReportColumn::new(name, label, role, None, ReportColumnData::String(values))
}

fn float_column(
    name: &'static str,
    label: &'static str,
    role: ColumnRole,
    values: Vec<Option<f64>>,
    decimals: u8,
) -> ReportColumn {
    ReportColumn::new(
        name,
        label,
        role,
        Some(DisplayFormat::new(DisplayUnit::Ratio, decimals)),
        ReportColumnData::Float64(values),
    )
}

fn count_column(name: &'static str, label: &'static str, values: Vec<Option<u64>>) -> ReportColumn {
    ReportColumn::new(
        name,
        label,
        ColumnRole::Measure,
        Some(DisplayFormat::new(DisplayUnit::Count, 0)),
        ReportColumnData::UInt64(values),
    )
}

fn datetime_column(
    name: &'static str,
    label: &'static str,
    values: Vec<Option<i64>>,
    time_unit: ReportTimeUnit,
    timezone: Option<String>,
) -> ReportColumn {
    ReportColumn::new(
        name,
        label,
        ColumnRole::Dimension,
        None,
        ReportColumnData::Datetime {
            values,
            time_unit,
            timezone,
        },
    )
}
