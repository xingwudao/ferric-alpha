use std::collections::BTreeMap;

use polars::prelude::DataFrame;

use crate::Result;
use crate::report::context::ReportContext;
use crate::report::{
    ColumnRole, DisplayFormat, DisplayUnit, ReportColumn, ReportColumnData, ReportTable,
};
use crate::statistics::sample_std;

pub fn factor_quantile_statistics(factor_data: &DataFrame) -> Result<ReportTable> {
    let context = ReportContext::new(factor_data)?;
    let mut observed = BTreeMap::<u32, Vec<f64>>::new();

    for row in &context.factor_data().rows {
        let Some(quantile) = row.quantile else {
            continue;
        };
        if quantile == 0 {
            continue;
        }
        let values = observed.entry(quantile).or_default();
        if let Some(factor) = row.factor {
            values.push(factor);
        }
    }

    let total = observed.values().map(Vec::len).sum::<usize>();
    let mut factor_quantile = Vec::with_capacity(observed.len());
    let mut factor_min = Vec::with_capacity(observed.len());
    let mut factor_max = Vec::with_capacity(observed.len());
    let mut factor_mean = Vec::with_capacity(observed.len());
    let mut factor_std = Vec::with_capacity(observed.len());
    let mut count = Vec::with_capacity(observed.len());
    let mut count_percent = Vec::with_capacity(observed.len());

    for (quantile, values) in observed {
        factor_quantile.push(Some(quantile));
        count.push(Some(values.len() as u64));

        if values.is_empty() {
            factor_min.push(None);
            factor_max.push(None);
            factor_mean.push(None);
            factor_std.push(None);
        } else {
            factor_min.push(Some(values.iter().copied().fold(f64::INFINITY, f64::min)));
            factor_max.push(Some(
                values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            ));
            factor_mean.push(Some(values.iter().sum::<f64>() / values.len() as f64));
            factor_std.push(sample_std(&values)?);
        }

        count_percent.push(if total == 0 {
            None
        } else {
            Some(values.len() as f64 / total as f64)
        });
    }

    ReportTable::new(
        "quantile.statistics",
        "Quantile Statistics",
        vec!["factor_quantile".to_string()],
        vec![
            ReportColumn::new(
                "factor_quantile",
                "Factor Quantile",
                ColumnRole::Dimension,
                Some(DisplayFormat::new(DisplayUnit::Count, 0)),
                ReportColumnData::UInt32(factor_quantile),
            ),
            ReportColumn::new(
                "factor_min",
                "Factor Min",
                ColumnRole::Statistic,
                Some(DisplayFormat::new(DisplayUnit::Raw, 6)),
                ReportColumnData::Float64(factor_min),
            ),
            ReportColumn::new(
                "factor_max",
                "Factor Max",
                ColumnRole::Statistic,
                Some(DisplayFormat::new(DisplayUnit::Raw, 6)),
                ReportColumnData::Float64(factor_max),
            ),
            ReportColumn::new(
                "factor_mean",
                "Factor Mean",
                ColumnRole::Statistic,
                Some(DisplayFormat::new(DisplayUnit::Raw, 6)),
                ReportColumnData::Float64(factor_mean),
            ),
            ReportColumn::new(
                "factor_std",
                "Factor Std",
                ColumnRole::Statistic,
                Some(DisplayFormat::new(DisplayUnit::Raw, 6)),
                ReportColumnData::Float64(factor_std),
            ),
            ReportColumn::new(
                "count",
                "Count",
                ColumnRole::Measure,
                Some(DisplayFormat::new(DisplayUnit::Count, 0)),
                ReportColumnData::UInt64(count),
            ),
            ReportColumn::new(
                "count_percent",
                "Count Percent",
                ColumnRole::Measure,
                Some(DisplayFormat::new(DisplayUnit::Percent, 4)),
                ReportColumnData::Float64(count_percent),
            ),
        ],
    )
}
