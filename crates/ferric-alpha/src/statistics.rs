use crate::{FerricAlphaError, Result};
use serde::{Deserialize, Serialize};
use statrs::distribution::{ContinuousCDF, Normal, StudentsT};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TTest {
    pub statistic: Option<f64>,
    pub p_value: Option<f64>,
    pub count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct QqPoint {
    pub probability: f64,
    pub theoretical: f64,
    pub observed: f64,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearFit {
    pub alpha: Option<f64>,
    pub beta: Option<f64>,
    pub count: usize,
}

pub fn average_rank(values: &[f64]) -> Result<Vec<f64>> {
    validate_finite("average_rank", values)?;

    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|left, right| left.1.total_cmp(&right.1));

    let mut ranks = vec![0.0; values.len()];
    let mut start = 0;
    while start < indexed.len() {
        let mut end = start + 1;
        while end < indexed.len() && indexed[end].1 == indexed[start].1 {
            end += 1;
        }

        let rank = ((start + 1) as f64 + end as f64) / 2.0;
        for (original_index, _) in &indexed[start..end] {
            ranks[*original_index] = rank;
        }
        start = end;
    }

    Ok(ranks)
}

pub fn pearson(x: &[f64], y: &[f64]) -> Result<Option<f64>> {
    validate_pair("pearson", x, y)?;
    pearson_validated(x, y)
}

pub fn spearman(x: &[f64], y: &[f64]) -> Result<Option<f64>> {
    validate_pair("spearman", x, y)?;
    let x_ranks = average_rank(x)?;
    let y_ranks = average_rank(y)?;
    pearson_validated(&x_ranks, &y_ranks)
}

pub fn simple_ols(x: &[f64], y: &[f64]) -> Result<LinearFit> {
    validate_pair("simple_ols", x, y)?;

    let count = x.len();
    if count < 2 {
        return Ok(LinearFit {
            alpha: None,
            beta: None,
            count,
        });
    }

    let x_mean = mean(x);
    let y_mean = mean(y);
    let mut sxx = 0.0;
    let mut sxy = 0.0;

    for (&x_value, &y_value) in x.iter().zip(y) {
        let x_delta = x_value - x_mean;
        sxx += x_delta * x_delta;
        sxy += x_delta * (y_value - y_mean);
    }

    if sxx == 0.0 {
        return Ok(LinearFit {
            alpha: None,
            beta: None,
            count,
        });
    }

    let beta = sxy / sxx;
    let alpha = y_mean - beta * x_mean;
    Ok(LinearFit {
        alpha: Some(alpha),
        beta: Some(beta),
        count,
    })
}

pub fn one_sample_t_test(values: &[f64], population_mean: f64) -> Result<TTest> {
    const OPERATION: &str = "one_sample_t_test";

    validate_finite(OPERATION, values)?;
    validate_finite_value(OPERATION, population_mean, "population mean must be finite")?;

    let count = values.len();
    if count < 2 {
        return Ok(TTest {
            statistic: None,
            p_value: None,
            count,
        });
    }

    let Some(std) = sample_std(values)? else {
        return Ok(TTest {
            statistic: None,
            p_value: None,
            count,
        });
    };

    if std == 0.0 {
        return Ok(TTest {
            statistic: None,
            p_value: None,
            count,
        });
    }

    let statistic = (mean(values) - population_mean) / (std / (count as f64).sqrt());
    let distribution = StudentsT::new(0.0, 1.0, (count - 1) as f64)
        .expect("degrees of freedom are positive for count >= 2");
    let p_value = 2.0 * distribution.sf(statistic.abs());

    Ok(TTest {
        statistic: Some(statistic),
        p_value: Some(p_value),
        count,
    })
}

pub fn biased_skew(values: &[f64]) -> Result<Option<f64>> {
    validate_finite("biased_skew", values)?;
    let Some((_, m2, m3, _)) = population_central_moments(values) else {
        return Ok(None);
    };

    if m2 == 0.0 {
        return Ok(None);
    }

    Ok(Some(m3 / m2.powf(1.5)))
}

pub fn biased_excess_kurtosis(values: &[f64]) -> Result<Option<f64>> {
    validate_finite("biased_excess_kurtosis", values)?;
    let Some((_, m2, _, m4)) = population_central_moments(values) else {
        return Ok(None);
    };

    if m2 == 0.0 {
        return Ok(None);
    }

    Ok(Some(m4 / m2.powi(2) - 3.0))
}

pub fn normal_qq(values: &[f64]) -> Result<Vec<QqPoint>> {
    const OPERATION: &str = "normal_qq";

    validate_finite(OPERATION, values)?;
    let count = values.len();
    if count < 2 {
        return Ok(Vec::new());
    }

    let Some(std) = sample_std(values)? else {
        return Ok(Vec::new());
    };

    if std == 0.0 {
        return Ok(Vec::new());
    }

    let value_mean = mean(values);
    let distribution = Normal::new(0.0, 1.0).expect("standard normal parameters are valid");
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);

    Ok(sorted
        .into_iter()
        .enumerate()
        .map(|(index, value)| {
            let probability = (index as f64 + 0.5) / count as f64;
            QqPoint {
                probability,
                theoretical: distribution.inverse_cdf(probability),
                observed: (value - value_mean) / std,
            }
        })
        .collect())
}

pub fn sample_std(values: &[f64]) -> Result<Option<f64>> {
    validate_finite("sample_std", values)?;

    if values.len() < 2 {
        return Ok(None);
    }

    let value_mean = mean(values);
    let sum_squares = values
        .iter()
        .map(|value| {
            let delta = value - value_mean;
            delta * delta
        })
        .sum::<f64>();

    Ok(Some((sum_squares / (values.len() - 1) as f64).sqrt()))
}

pub fn std_error(values: &[f64]) -> Result<Option<f64>> {
    Ok(sample_std(values)?.map(|std| std / (values.len() as f64).sqrt()))
}

fn pearson_validated(x: &[f64], y: &[f64]) -> Result<Option<f64>> {
    if x.len() < 2 {
        return Ok(None);
    }

    let x_mean = mean(x);
    let y_mean = mean(y);
    let mut sxx = 0.0;
    let mut syy = 0.0;
    let mut sxy = 0.0;

    for (&x_value, &y_value) in x.iter().zip(y) {
        let x_delta = x_value - x_mean;
        let y_delta = y_value - y_mean;
        sxx += x_delta * x_delta;
        syy += y_delta * y_delta;
        sxy += x_delta * y_delta;
    }

    if sxx == 0.0 || syy == 0.0 {
        return Ok(None);
    }

    Ok(Some((sxy / (sxx * syy).sqrt()).clamp(-1.0, 1.0)))
}

fn validate_pair(operation: &'static str, x: &[f64], y: &[f64]) -> Result<()> {
    if x.len() != y.len() {
        return Err(FerricAlphaError::InvalidInput {
            operation,
            reason: "paired inputs must have equal length",
        });
    }

    validate_finite(operation, x)?;
    validate_finite(operation, y)
}

fn validate_finite(operation: &'static str, values: &[f64]) -> Result<()> {
    if values.iter().all(|value| value.is_finite()) {
        return Ok(());
    }

    Err(FerricAlphaError::InvalidInput {
        operation,
        reason: "values must be finite",
    })
}

fn validate_finite_value(operation: &'static str, value: f64, reason: &'static str) -> Result<()> {
    if value.is_finite() {
        return Ok(());
    }

    Err(FerricAlphaError::InvalidInput { operation, reason })
}

fn population_central_moments(values: &[f64]) -> Option<(f64, f64, f64, f64)> {
    if values.is_empty() {
        return None;
    }

    let value_mean = mean(values);
    let count = values.len() as f64;
    let mut m2 = 0.0;
    let mut m3 = 0.0;
    let mut m4 = 0.0;

    for value in values {
        let delta = value - value_mean;
        let delta2 = delta * delta;
        m2 += delta2;
        m3 += delta2 * delta;
        m4 += delta2 * delta2;
    }

    Some((value_mean, m2 / count, m3 / count, m4 / count))
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}
