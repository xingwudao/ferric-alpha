use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use polars::prelude::{DataFrame, DataType, IntoColumn, NamedFrom, Series, SortMultipleOptions};
use serde::{Deserialize, Serialize};

use crate::data::{ensure_matching_date_timezone, required_column};
use crate::error::{FerricAlphaError, Result};
use crate::performance::frame::{ensure_dtype, validate_factor_data};
use crate::statistics::sample_std;

const OPERATION: &str = "average_cumulative_return_by_quantile";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EventReturnsOptions {
    pub periods_before: u32,
    pub periods_after: u32,
    pub demeaned: bool,
    pub group_adjust: bool,
    pub by_group: bool,
}

impl Default for EventReturnsOptions {
    fn default() -> Self {
        Self {
            periods_before: 10,
            periods_after: 15,
            demeaned: true,
            group_adjust: false,
            by_group: false,
        }
    }
}

impl EventReturnsOptions {
    pub fn with_periods_before(mut self, periods_before: u32) -> Self {
        self.periods_before = periods_before;
        self
    }

    pub fn with_periods_after(mut self, periods_after: u32) -> Self {
        self.periods_after = periods_after;
        self
    }

    pub fn with_demeaned(mut self, demeaned: bool) -> Self {
        self.demeaned = demeaned;
        self
    }

    pub fn with_group_adjust(mut self, group_adjust: bool) -> Self {
        self.group_adjust = group_adjust;
        self
    }

    pub fn with_by_group(mut self, by_group: bool) -> Self {
        self.by_group = by_group;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
#[allow(dead_code)]
pub(crate) struct EventCoverageRow {
    pub(crate) scope: String,
    pub(crate) offset: Option<i32>,
    pub(crate) input_events: u64,
    pub(crate) used_events: u64,
    pub(crate) skipped_missing_date: u64,
    pub(crate) imputed_return_cells: u64,
    pub(crate) path_count: u64,
}

#[derive(Debug, Clone)]
pub(crate) struct EventWindowResult {
    pub(crate) frame: DataFrame,
    #[allow(dead_code)]
    pub(crate) coverage: Vec<EventCoverageRow>,
}

#[derive(Debug, Clone)]
struct Event {
    date: i64,
    session_index: usize,
    asset: String,
    group: Option<String>,
    quantile: u32,
}

#[derive(Debug, Clone)]
struct ReturnCalendar {
    dates: Vec<i64>,
    values: HashMap<(String, i64), Option<f64>>,
}

#[derive(Debug, Clone, Default)]
struct OffsetCoverage {
    used_events: u64,
    imputed_return_cells: u64,
    path_count: u64,
}

pub fn average_cumulative_return_by_quantile(
    factor_data: &DataFrame,
    returns: &DataFrame,
    options: EventReturnsOptions,
) -> Result<DataFrame> {
    Ok(event_window_result(factor_data, returns, options)?.frame)
}

pub(crate) fn event_window_result(
    factor_data: &DataFrame,
    returns: &DataFrame,
    options: EventReturnsOptions,
) -> Result<EventWindowResult> {
    let data = validate_factor_data(
        factor_data,
        false,
        options.group_adjust || options.by_group,
        false,
    )?;
    let quantile = required_column(factor_data, "factor_quantile")?;
    ensure_dtype("factor_quantile", quantile.dtype(), "UInt32", |dtype| {
        matches!(dtype, DataType::UInt32)
    })?;

    let calendar = validate_returns(factor_data, returns)?;
    if calendar.dates.is_empty() {
        return Err(FerricAlphaError::InsufficientData {
            operation: OPERATION,
        });
    }
    let date_index = calendar
        .dates
        .iter()
        .enumerate()
        .map(|(index, date)| (*date, index))
        .collect::<HashMap<_, _>>();

    let mut input_events = 0_u64;
    let mut skipped_missing_date = 0_u64;
    let mut events = Vec::new();
    for row in &data.rows {
        let Some(quantile) = row.quantile.filter(|quantile| *quantile > 0) else {
            continue;
        };
        if (options.group_adjust || options.by_group) && row.group.is_none() {
            return Err(FerricAlphaError::InvalidInput {
                operation: OPERATION,
                reason: "group must not contain null values",
            });
        }
        input_events += 1;
        let Some(session_index) = date_index.get(&row.date).copied() else {
            skipped_missing_date += 1;
            continue;
        };
        events.push(Event {
            date: row.date,
            session_index,
            asset: row.asset.clone(),
            group: row.group.clone(),
            quantile,
        });
    }

    if events.is_empty() {
        return Err(FerricAlphaError::InsufficientData {
            operation: OPERATION,
        });
    }

    let event_dates = events
        .iter()
        .map(|event| event.date)
        .collect::<BTreeSet<_>>();
    let mut assets = events
        .iter()
        .map(|event| event.asset.clone())
        .collect::<BTreeSet<_>>();
    if options.demeaned || options.group_adjust {
        for row in &data.rows {
            if event_dates.contains(&row.date) {
                assets.insert(row.asset.clone());
            }
        }
    }

    let (wealth, densify_imputations) = cumulative_wealth(&calendar, &assets);
    let by_event_date = group_events_by_date_partition(&events, options.by_group);
    let universe_by_date = universe_by_date(&data.rows, &event_dates);
    let universe_by_date_group = universe_by_date_group(&data.rows, &event_dates);

    let mut paths: BTreeMap<(u32, Option<String>, i32), Vec<f64>> = BTreeMap::new();
    let mut coverage_by_offset: BTreeMap<i32, OffsetCoverage> = BTreeMap::new();
    let start = -(options.periods_before as i32);
    let end = options.periods_after as i32;
    for event_offset in start..=end {
        coverage_by_offset.entry(event_offset).or_default();
    }

    for ((event_date, quantile, group), partition_events) in by_event_date {
        let session_index = partition_events[0].session_index;
        for offset in start..=end {
            let target_index = session_index as i32 + offset;
            if target_index < 0 || target_index >= calendar.dates.len() as i32 {
                continue;
            }
            let target_index = target_index as usize;
            let mut values = Vec::with_capacity(partition_events.len());
            let mut imputed = 0_u64;
            for event in &partition_events {
                let (mut value, was_imputed) =
                    rebased_wealth_value(&wealth, &event.asset, event.session_index, target_index);
                if options.group_adjust {
                    let group_assets = event
                        .group
                        .as_ref()
                        .and_then(|group| universe_by_date_group.get(&(event_date, group.clone())));
                    let (group_mean, group_imputed) = mean_rebased_wealth_with_imputations(
                        group_assets,
                        &wealth,
                        session_index,
                        target_index,
                    );
                    value -= group_mean.unwrap_or(0.0);
                    imputed += group_imputed;
                }
                values.push(value);
                imputed += u64::from(was_imputed);
            }
            let Some(mut event_path) = mean(&values) else {
                continue;
            };

            if options.demeaned && !options.group_adjust {
                let (universe_mean, universe_imputed) = mean_rebased_wealth_with_imputations(
                    universe_by_date.get(&event_date),
                    &wealth,
                    session_index,
                    target_index,
                );
                event_path -= universe_mean.unwrap_or(0.0);
                imputed += universe_imputed;
            }

            let key = (quantile, group.clone().filter(|_| options.by_group), offset);
            paths.entry(key).or_default().push(event_path);
            let offset_coverage = coverage_by_offset.entry(offset).or_default();
            offset_coverage.used_events += partition_events.len() as u64;
            offset_coverage.imputed_return_cells += imputed;
            offset_coverage.path_count += 1;
        }
    }

    let mut factor_quantiles = Vec::new();
    let mut groups = Vec::new();
    let mut offsets = Vec::new();
    let mut means = Vec::new();
    let mut stds = Vec::new();
    let mut counts = Vec::new();
    for ((quantile, group, offset), values) in paths {
        factor_quantiles.push(quantile);
        if options.by_group {
            groups.push(group.expect("grouped output should carry a group"));
        }
        offsets.push(offset);
        means.push(mean(&values));
        stds.push(sample_std(&values)?);
        counts.push(values.len() as u64);
    }

    let mut columns = vec![
        Series::new("factor_quantile".into(), factor_quantiles).into_column(),
        Series::new("offset".into(), offsets).into_column(),
        Series::new("mean_cumulative_return".into(), means).into_column(),
        Series::new("std_cumulative_return".into(), stds).into_column(),
        Series::new("count".into(), counts).into_column(),
    ];
    let sort = if options.by_group {
        columns.insert(1, Series::new("group".into(), groups).into_column());
        vec!["factor_quantile", "group", "offset"]
    } else {
        vec!["factor_quantile", "offset"]
    };
    let frame =
        DataFrame::new(columns[0].len(), columns)?.sort(sort, SortMultipleOptions::default())?;

    let mut coverage = Vec::with_capacity(coverage_by_offset.len() + 1);
    coverage.push(EventCoverageRow {
        scope: "overall".to_owned(),
        offset: None,
        input_events,
        used_events: events.len() as u64,
        skipped_missing_date,
        imputed_return_cells: densify_imputations,
        path_count: 0,
    });
    for (offset, row) in coverage_by_offset {
        coverage.push(EventCoverageRow {
            scope: "offset".to_owned(),
            offset: Some(offset),
            input_events,
            used_events: row.used_events,
            skipped_missing_date,
            imputed_return_cells: row.imputed_return_cells,
            path_count: row.path_count,
        });
    }

    Ok(EventWindowResult { frame, coverage })
}

fn validate_returns(factor_data: &DataFrame, returns: &DataFrame) -> Result<ReturnCalendar> {
    ensure_matching_date_timezone(factor_data, returns)?;
    let date = required_column(returns, "date")?;
    let asset = required_column(returns, "asset")?;
    let ret = required_column(returns, "return")?;
    ensure_dtype("asset", asset.dtype(), "String", |dtype| {
        matches!(dtype, DataType::String)
    })?;
    ensure_dtype("return", ret.dtype(), "Float64", |dtype| {
        matches!(dtype, DataType::Float64)
    })?;
    if date.null_count() > 0 || asset.null_count() > 0 {
        return Err(FerricAlphaError::NullKey);
    }

    let dates = date.as_materialized_series().datetime()?;
    let assets = asset.as_materialized_series().str()?;
    let returns_values = ret.as_materialized_series().f64()?;
    let mut seen = HashSet::with_capacity(returns.height());
    let mut calendar_dates = BTreeSet::new();
    let mut values = HashMap::with_capacity(returns.height());
    for ((date, asset), ret) in dates
        .physical()
        .iter()
        .zip(assets.iter())
        .zip(returns_values.iter())
    {
        let (Some(date), Some(asset)) = (date, asset) else {
            return Err(FerricAlphaError::NullKey);
        };
        if !seen.insert((date, asset.to_owned())) {
            return Err(FerricAlphaError::DuplicateKey);
        }
        calendar_dates.insert(date);
        values.insert(
            (asset.to_owned(), date),
            ret.filter(|value| value.is_finite()),
        );
    }

    Ok(ReturnCalendar {
        dates: calendar_dates.into_iter().collect(),
        values,
    })
}

fn cumulative_wealth(
    calendar: &ReturnCalendar,
    assets: &BTreeSet<String>,
) -> (HashMap<String, Vec<(f64, bool)>>, u64) {
    let mut wealth = HashMap::with_capacity(assets.len());
    let mut imputed = 0_u64;
    for asset in assets {
        let mut cumulative = 1.0;
        let mut values = Vec::with_capacity(calendar.dates.len());
        for date in &calendar.dates {
            let return_value = calendar
                .values
                .get(&(asset.clone(), *date))
                .copied()
                .flatten();
            let was_imputed = return_value.is_none();
            if was_imputed {
                imputed += 1;
            }
            cumulative *= 1.0 + return_value.unwrap_or(0.0);
            values.push((cumulative - 1.0, was_imputed));
        }
        wealth.insert(asset.clone(), values);
    }
    (wealth, imputed)
}

fn group_events_by_date_partition(
    events: &[Event],
    by_group: bool,
) -> BTreeMap<(i64, u32, Option<String>), Vec<Event>> {
    let mut grouped = BTreeMap::new();
    for event in events {
        grouped
            .entry((
                event.date,
                event.quantile,
                event.group.clone().filter(|_| by_group),
            ))
            .or_insert_with(Vec::new)
            .push(event.clone());
    }
    grouped
}

fn universe_by_date(
    rows: &[crate::performance::frame::FactorRow],
    event_dates: &BTreeSet<i64>,
) -> BTreeMap<i64, Vec<String>> {
    let mut universes: BTreeMap<i64, BTreeSet<String>> = BTreeMap::new();
    for row in rows {
        if event_dates.contains(&row.date) {
            universes
                .entry(row.date)
                .or_default()
                .insert(row.asset.clone());
        }
    }
    universes
        .into_iter()
        .map(|(date, assets)| (date, assets.into_iter().collect()))
        .collect()
}

fn universe_by_date_group(
    rows: &[crate::performance::frame::FactorRow],
    event_dates: &BTreeSet<i64>,
) -> BTreeMap<(i64, String), Vec<String>> {
    let mut universes: BTreeMap<(i64, String), BTreeSet<String>> = BTreeMap::new();
    for row in rows {
        if event_dates.contains(&row.date)
            && let Some(group) = &row.group
        {
            universes
                .entry((row.date, group.clone()))
                .or_default()
                .insert(row.asset.clone());
        }
    }
    universes
        .into_iter()
        .map(|(key, assets)| (key, assets.into_iter().collect()))
        .collect()
}

fn mean_rebased_wealth_with_imputations(
    assets: Option<&Vec<String>>,
    wealth: &HashMap<String, Vec<(f64, bool)>>,
    base_index: usize,
    target_index: usize,
) -> (Option<f64>, u64) {
    let Some(assets) = assets else {
        return (None, 0);
    };
    let mut imputed = 0_u64;
    let values = assets
        .iter()
        .map(|asset| {
            let (value, was_imputed) =
                rebased_wealth_value(wealth, asset, base_index, target_index);
            imputed += u64::from(was_imputed);
            value
        })
        .collect::<Vec<_>>();
    (mean(&values), imputed)
}

fn wealth_value(
    wealth: &HashMap<String, Vec<(f64, bool)>>,
    asset: &str,
    target_index: usize,
) -> (f64, bool) {
    wealth
        .get(asset)
        .map(|path| path[target_index])
        .unwrap_or((0.0, true))
}

fn rebased_wealth_value(
    wealth: &HashMap<String, Vec<(f64, bool)>>,
    asset: &str,
    base_index: usize,
    target_index: usize,
) -> (f64, bool) {
    let (target, was_imputed) = wealth_value(wealth, asset, target_index);
    let (base, _) = wealth_value(wealth, asset, base_index);
    (((1.0 + target) / (1.0 + base)) - 1.0, was_imputed)
}

fn mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        None
    } else {
        Some(values.iter().sum::<f64>() / values.len() as f64)
    }
}

#[cfg(test)]
mod tests {
    use polars::prelude::{DataFrame, DataType, IntoColumn, NamedFrom, Series, TimeUnit};

    use super::{EventReturnsOptions, event_window_result};

    const JAN_1: i64 = 1_704_067_200_000;
    const DAY_MS: i64 = 86_400_000;

    fn datetime_column(values: &[i64]) -> polars::prelude::Column {
        Series::new("date".into(), values)
            .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
            .expect("test dates should cast")
            .into_column()
    }

    #[test]
    fn event_window_result_carries_overall_and_offset_coverage() {
        let jan_2 = JAN_1 + DAY_MS;
        let jan_3 = jan_2 + DAY_MS;
        let absent = jan_3 + DAY_MS;
        let factor_data = DataFrame::new(
            3,
            vec![
                datetime_column(&[JAN_1, jan_2, absent]),
                Series::new("asset".into(), &["A", "A", "A"]).into_column(),
                Series::new("factor".into(), &[Some(1.0); 3]).into_column(),
                Series::new("factor_quantile".into(), &[Some(1_u32); 3]).into_column(),
            ],
        )
        .expect("test factor data should build");
        let returns = DataFrame::new(
            3,
            vec![
                datetime_column(&[JAN_1, jan_2, jan_3]),
                Series::new("asset".into(), &["A", "A", "A"]).into_column(),
                Series::new("return".into(), &[Some(0.10), None, Some(0.10)]).into_column(),
            ],
        )
        .expect("test returns should build");

        let result = event_window_result(
            &factor_data,
            &returns,
            EventReturnsOptions {
                periods_before: 1,
                periods_after: 1,
                demeaned: false,
                group_adjust: false,
                by_group: false,
            },
        )
        .unwrap();

        assert_eq!(result.coverage[0].scope, "overall");
        assert_eq!(result.coverage[0].offset, None);
        assert_eq!(result.coverage[0].input_events, 3);
        assert_eq!(result.coverage[0].used_events, 2);
        assert_eq!(result.coverage[0].skipped_missing_date, 1);
        assert_eq!(result.coverage[0].imputed_return_cells, 1);

        let offset_zero = result
            .coverage
            .iter()
            .find(|row| row.offset == Some(0))
            .expect("offset zero coverage should exist");
        assert_eq!(offset_zero.scope, "offset");
        assert_eq!(offset_zero.used_events, 2);
        assert_eq!(offset_zero.imputed_return_cells, 1);
        assert_eq!(offset_zero.path_count, 2);
    }

    #[test]
    fn event_window_coverage_counts_raw_events_and_adjustment_imputations() {
        let factor_data = DataFrame::new(
            2,
            vec![
                datetime_column(&[JAN_1, JAN_1]),
                Series::new("asset".into(), &["A", "B"]).into_column(),
                Series::new("group".into(), &["g1", "g1"]).into_column(),
                Series::new("factor".into(), &[Some(1.0); 2]).into_column(),
                Series::new("factor_quantile".into(), &[Some(1_u32); 2]).into_column(),
            ],
        )
        .expect("test factor data should build");
        let returns = DataFrame::new(
            2,
            vec![
                datetime_column(&[JAN_1, JAN_1]),
                Series::new("asset".into(), &["A", "B"]).into_column(),
                Series::new("return".into(), &[Some(0.10), None]).into_column(),
            ],
        )
        .expect("test returns should build");

        let result = event_window_result(
            &factor_data,
            &returns,
            EventReturnsOptions {
                periods_before: 0,
                periods_after: 0,
                demeaned: false,
                group_adjust: true,
                by_group: false,
            },
        )
        .unwrap();

        let offset_zero = result
            .coverage
            .iter()
            .find(|row| row.offset == Some(0))
            .expect("offset zero coverage should exist");
        assert_eq!(offset_zero.used_events, 2);
        assert_eq!(offset_zero.imputed_return_cells, 3);
        assert_eq!(offset_zero.path_count, 1);
    }

    #[test]
    fn event_window_rebases_each_path_to_the_event_session() {
        let jan_2 = JAN_1 + DAY_MS;
        let jan_3 = jan_2 + DAY_MS;
        let factor_data = DataFrame::new(
            1,
            vec![
                datetime_column(&[jan_2]),
                Series::new("asset".into(), &["A"]).into_column(),
                Series::new("factor".into(), &[Some(1.0)]).into_column(),
                Series::new("factor_quantile".into(), &[Some(1_u32)]).into_column(),
            ],
        )
        .expect("test factor data should build");
        let returns = DataFrame::new(
            3,
            vec![
                datetime_column(&[JAN_1, jan_2, jan_3]),
                Series::new("asset".into(), &["A", "A", "A"]).into_column(),
                Series::new("return".into(), &[Some(0.10), Some(0.10), Some(0.10)]).into_column(),
            ],
        )
        .expect("test returns should build");

        let result = event_window_result(
            &factor_data,
            &returns,
            EventReturnsOptions {
                periods_before: 1,
                periods_after: 1,
                demeaned: false,
                group_adjust: false,
                by_group: false,
            },
        )
        .unwrap();
        let values = result
            .frame
            .column("mean_cumulative_return")
            .unwrap()
            .as_materialized_series()
            .f64()
            .unwrap()
            .iter()
            .collect::<Vec<_>>();

        assert_eq!(values.len(), 3);
        approx::assert_abs_diff_eq!(values[0].unwrap(), -0.090909090909, epsilon = 1e-12);
        approx::assert_abs_diff_eq!(values[1].unwrap(), 0.0, epsilon = 1e-12);
        approx::assert_abs_diff_eq!(values[2].unwrap(), 0.10, epsilon = 1e-12);
    }
}
