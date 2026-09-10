use std::collections::BTreeMap;

use polars::prelude::{DataFrame, NamedFrom, Series, SortMultipleOptions};

use crate::data::validate_factor_frame;
use crate::error::{FerricAlphaError, Result};

#[derive(Debug, Clone, PartialEq)]
pub struct QuantizeOptions {
    quantiles: Option<u32>,
    quantile_edges: Option<Vec<f64>>,
    bin_count: Option<u32>,
    bins: Option<Vec<f64>>,
    by_group: bool,
    zero_aware: bool,
}

impl QuantizeOptions {
    pub fn quantiles(quantiles: u32) -> Self {
        Self {
            quantiles: Some(quantiles),
            quantile_edges: None,
            bin_count: None,
            bins: None,
            by_group: false,
            zero_aware: false,
        }
    }

    pub fn quantile_edges(edges: Vec<f64>) -> Self {
        Self {
            quantiles: None,
            quantile_edges: Some(edges),
            bin_count: None,
            bins: None,
            by_group: false,
            zero_aware: false,
        }
    }

    pub fn bin_count(bins: u32) -> Self {
        Self {
            quantiles: None,
            quantile_edges: None,
            bin_count: Some(bins),
            bins: None,
            by_group: false,
            zero_aware: false,
        }
    }

    pub fn bins(bins: Vec<f64>) -> Self {
        Self {
            quantiles: None,
            quantile_edges: None,
            bin_count: None,
            bins: Some(bins),
            by_group: false,
            zero_aware: false,
        }
    }

    pub fn with_by_group(mut self, by_group: bool) -> Self {
        self.by_group = by_group;
        self
    }

    pub fn with_zero_aware(mut self, zero_aware: bool) -> Self {
        self.zero_aware = zero_aware;
        self
    }

    fn mode(&self) -> Result<QuantizeMode<'_>> {
        let active_modes = u8::from(self.quantiles.is_some())
            + u8::from(self.quantile_edges.is_some())
            + u8::from(self.bin_count.is_some())
            + u8::from(self.bins.is_some());
        if active_modes > 1 {
            return Err(FerricAlphaError::ConflictingQuantileMode);
        }
        if let Some(quantiles) = self.quantiles {
            if quantiles < 2 {
                return Err(FerricAlphaError::InvalidQuantiles {
                    reason: "quantiles must be at least 2",
                });
            }
            if self.zero_aware && quantiles % 2 != 0 {
                return Err(FerricAlphaError::InvalidQuantiles {
                    reason: "zero-aware quantiles must be even",
                });
            }
            return Ok(QuantizeMode::Quantiles(quantiles));
        }

        if let Some(edges) = self.quantile_edges.as_deref() {
            if edges.len() < 2 {
                return Err(FerricAlphaError::InvalidQuantiles {
                    reason: "quantile edges must contain at least two boundaries",
                });
            }
            for edge in edges {
                if !edge.is_finite() || *edge < 0.0 || *edge > 1.0 {
                    return Err(FerricAlphaError::InvalidQuantiles {
                        reason: "quantile edges must be finite values in [0, 1]",
                    });
                }
            }
            for pair in edges.windows(2) {
                if pair[0] >= pair[1] {
                    return Err(FerricAlphaError::InvalidQuantiles {
                        reason: "quantile edges must be strictly increasing",
                    });
                }
            }
            return Ok(QuantizeMode::QuantileEdges(edges));
        }

        if let Some(bin_count) = self.bin_count {
            if bin_count < 1 {
                return Err(FerricAlphaError::InvalidQuantiles {
                    reason: "bins must be at least 1",
                });
            }
            if self.zero_aware && bin_count % 2 != 0 {
                return Err(FerricAlphaError::InvalidQuantiles {
                    reason: "zero-aware bins must be even",
                });
            }
            return Ok(QuantizeMode::BinCount(bin_count));
        }

        let bins = self
            .bins
            .as_deref()
            .ok_or(FerricAlphaError::InvalidQuantiles {
                reason: "quantiles or bins are required",
            })?;
        if bins.len() < 2 {
            return Err(FerricAlphaError::InvalidQuantiles {
                reason: "bins must contain at least two boundaries",
            });
        }
        if self.zero_aware {
            return Err(FerricAlphaError::InvalidQuantiles {
                reason: "zero-aware explicit bins are not supported in this phase",
            });
        }
        for pair in bins.windows(2) {
            if !pair[0].is_finite() || !pair[1].is_finite() || pair[0] >= pair[1] {
                return Err(FerricAlphaError::InvalidQuantiles {
                    reason: "bin boundaries must be finite and strictly increasing",
                });
            }
        }
        Ok(QuantizeMode::Bins(bins))
    }
}

#[derive(Debug, Clone, Copy)]
enum QuantizeMode<'a> {
    Quantiles(u32),
    QuantileEdges(&'a [f64]),
    BinCount(u32),
    Bins(&'a [f64]),
}

pub fn quantize_factor(factor: &DataFrame, options: QuantizeOptions) -> Result<DataFrame> {
    let mode = options.mode()?;
    let factor = validate_factor_frame(factor)?;
    if options.by_group && factor.column("group").is_err() {
        return Err(FerricAlphaError::MissingColumn("group"));
    }

    let dates = factor.column("date")?.as_materialized_series().datetime()?;
    let assets = factor.column("asset")?.as_materialized_series().str()?;
    let factors = factor.column("factor")?.as_materialized_series().f64()?;
    let groups = if options.by_group {
        Some(factor.column("group")?.as_materialized_series().str()?)
    } else {
        None
    };

    let mut rows = Vec::with_capacity(factor.height());
    let mut partitions: BTreeMap<(i64, Option<String>), Vec<usize>> = BTreeMap::new();
    for index in 0..factor.height() {
        let date = dates
            .physical()
            .get(index)
            .expect("validated factor date should not be null");
        let asset = assets
            .get(index)
            .expect("validated factor asset should not be null")
            .to_owned();
        let group = groups
            .as_ref()
            .and_then(|groups| groups.get(index).map(ToOwned::to_owned));
        let factor_value = factors.get(index).filter(|value| value.is_finite());
        rows.push(QuantileRow {
            asset,
            group: group.clone(),
            factor: factor_value,
        });
        if factor_value.is_some() {
            partitions.entry((date, group)).or_default().push(index);
        }
    }

    let mut assignments = vec![None; rows.len()];
    for indexes in partitions.values() {
        match mode {
            QuantizeMode::Quantiles(quantiles) if options.zero_aware => {
                assign_zero_aware(indexes, &rows, quantiles, &mut assignments);
            }
            QuantizeMode::Quantiles(quantiles) => {
                assign_quantiles(indexes, &rows, quantiles, 1, &mut assignments);
            }
            QuantizeMode::QuantileEdges(edges) => {
                if options.zero_aware {
                    assign_zero_aware_edges(indexes, &rows, edges, &mut assignments);
                } else {
                    assign_quantile_edges(indexes, &rows, edges, 1, &mut assignments);
                }
            }
            QuantizeMode::BinCount(bin_count) if options.zero_aware => {
                assign_zero_aware_bin_count(indexes, &rows, bin_count, &mut assignments);
            }
            QuantizeMode::BinCount(bin_count) => {
                assign_bin_count(indexes, &rows, bin_count, 1, &mut assignments);
            }
            QuantizeMode::Bins(bins) => {
                assign_bins(indexes, &rows, bins, false, &mut assignments);
            }
        }
    }

    let mut columns = factor.columns().to_vec();
    columns.push(Series::new("factor_quantile".into(), assignments).into());
    Ok(DataFrame::new(factor.height(), columns)?
        .sort(["date", "asset"], SortMultipleOptions::default())?)
}

fn assign_bins(
    indexes: &[usize],
    rows: &[QuantileRow],
    bins: &[f64],
    include_lowest: bool,
    assignments: &mut [Option<u32>],
) {
    for index in indexes {
        let value = rows[*index].factor.expect("valid row should have factor");
        let bucket = bins
            .windows(2)
            .enumerate()
            .position(|(bucket, pair)| {
                (value > pair[0] || (include_lowest && bucket == 0 && value == pair[0]))
                    && value <= pair[1]
            })
            .map(|bucket| bucket as u32 + 1);
        assignments[*index] = bucket;
    }
}

fn assign_bin_count(
    indexes: &[usize],
    rows: &[QuantileRow],
    bin_count: u32,
    offset: u32,
    assignments: &mut [Option<u32>],
) {
    if indexes.is_empty() {
        return;
    }
    let values = indexes
        .iter()
        .map(|index| rows[*index].factor.expect("valid row should have factor"))
        .collect::<Vec<_>>();
    let min = values.iter().copied().reduce(f64::min).unwrap();
    let max = values.iter().copied().reduce(f64::max).unwrap();
    if bin_count == 1 || min == max {
        for index in indexes {
            assignments[*index] = Some(offset);
        }
        return;
    }
    let width = (max - min) / bin_count as f64;
    for index in indexes {
        let value = rows[*index].factor.expect("valid row should have factor");
        let bucket = if value == min {
            1
        } else {
            ((value - min) / width).ceil().clamp(1.0, bin_count as f64) as u32
        };
        assignments[*index] = Some(offset + bucket - 1);
    }
}

#[derive(Debug)]
struct QuantileRow {
    asset: String,
    group: Option<String>,
    factor: Option<f64>,
}

fn assign_zero_aware(
    indexes: &[usize],
    rows: &[QuantileRow],
    quantiles: u32,
    assignments: &mut [Option<u32>],
) {
    let midpoint = quantiles / 2;
    let negative = indexes
        .iter()
        .copied()
        .filter(|index| rows[*index].factor.expect("valid row should have factor") < 0.0)
        .collect::<Vec<_>>();
    let positive = indexes
        .iter()
        .copied()
        .filter(|index| rows[*index].factor.expect("valid row should have factor") >= 0.0)
        .collect::<Vec<_>>();
    assign_quantiles(&negative, rows, midpoint, 1, assignments);
    assign_quantiles(&positive, rows, midpoint, midpoint + 1, assignments);
}

fn assign_zero_aware_edges(
    indexes: &[usize],
    rows: &[QuantileRow],
    edges: &[f64],
    assignments: &mut [Option<u32>],
) {
    let bucket_count = edges.len() as u32 - 1;
    let midpoint = bucket_count / 2;
    let negative = indexes
        .iter()
        .copied()
        .filter(|index| rows[*index].factor.expect("valid row should have factor") < 0.0)
        .collect::<Vec<_>>();
    let positive = indexes
        .iter()
        .copied()
        .filter(|index| rows[*index].factor.expect("valid row should have factor") >= 0.0)
        .collect::<Vec<_>>();
    assign_quantile_edges(&negative, rows, edges, 1, assignments);
    assign_quantile_edges(&positive, rows, edges, midpoint + 1, assignments);
}

fn assign_zero_aware_bin_count(
    indexes: &[usize],
    rows: &[QuantileRow],
    bin_count: u32,
    assignments: &mut [Option<u32>],
) {
    let midpoint = bin_count / 2;
    let negative = indexes
        .iter()
        .copied()
        .filter(|index| rows[*index].factor.expect("valid row should have factor") < 0.0)
        .collect::<Vec<_>>();
    let positive = indexes
        .iter()
        .copied()
        .filter(|index| rows[*index].factor.expect("valid row should have factor") >= 0.0)
        .collect::<Vec<_>>();
    assign_bin_count(&negative, rows, midpoint, 1, assignments);
    assign_bin_count(&positive, rows, midpoint, midpoint + 1, assignments);
}

fn assign_quantile_edges(
    indexes: &[usize],
    rows: &[QuantileRow],
    edges: &[f64],
    offset: u32,
    assignments: &mut [Option<u32>],
) {
    if indexes.is_empty() {
        return;
    }

    let mut values = indexes
        .iter()
        .map(|index| rows[*index].factor.expect("valid row should have factor"))
        .collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    let cut_points = edges
        .iter()
        .map(|edge| percentile(&values, *edge))
        .collect::<Vec<_>>();
    assign_bins(indexes, rows, &cut_points, true, assignments);
    for index in indexes {
        if let Some(value) = &mut assignments[*index]
            && *value > 0
        {
            *value += offset - 1;
        }
    }
}

fn assign_quantiles(
    indexes: &[usize],
    rows: &[QuantileRow],
    quantiles: u32,
    offset: u32,
    assignments: &mut [Option<u32>],
) {
    if indexes.is_empty() {
        return;
    }
    if indexes.len() < 2 {
        return;
    }

    let mut sorted = indexes.to_vec();
    sorted.sort_by(|left, right| {
        let left_row = &rows[*left];
        let right_row = &rows[*right];
        left_row
            .factor
            .partial_cmp(&right_row.factor)
            .expect("finite factors should be comparable")
            .then_with(|| left_row.group.cmp(&right_row.group))
            .then_with(|| left_row.asset.cmp(&right_row.asset))
    });

    for (rank, index) in sorted.iter().enumerate() {
        let bucket = if sorted.len() >= quantiles as usize {
            ((rank as u32 * quantiles) / sorted.len() as u32).min(quantiles - 1)
        } else {
            let scaled_rank = rank as f64 * (quantiles - 1) as f64 / (sorted.len() - 1) as f64;
            scaled_rank.round() as u32
        };
        assignments[*index] = Some(offset + bucket);
    }
}

fn percentile(sorted: &[f64], probability: f64) -> f64 {
    if sorted.len() == 1 {
        return sorted[0];
    }
    let position = probability * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        sorted[lower]
    } else {
        let weight = position - lower as f64;
        sorted[lower] * (1.0 - weight) + sorted[upper] * weight
    }
}
