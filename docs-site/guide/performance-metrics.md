# Performance Metrics

Ferric Alpha implements the core factor-analysis metrics expected by
alphalens users while exposing Polars DataFrames at the Python boundary.

## Information Coefficient

```python
ic = fa.performance.factor_information_coefficient(factor_data)
mean_ic = fa.performance.mean_information_coefficient(factor_data)
```

IC is computed as cross-sectional Spearman rank correlation between factor
values and forward returns.

## Factor Portfolio

```python
weights = fa.performance.factor_weights(factor_data)
returns = fa.performance.factor_returns(factor_data)
alpha_beta = fa.performance.factor_alpha_beta(factor_data)
```

Weights can be demeaned, group-adjusted, or equal-weighted. Factor returns use
the same weight construction options.

## Quantiles And Turnover

```python
quantile_returns = fa.performance.mean_return_by_quantile(factor_data)
spread = fa.performance.compute_mean_returns_spread(quantile_returns, 5, 1)
turnover = fa.performance.quantile_turnover(factor_data, quantile=5, period=1)
rank_auto = fa.performance.factor_rank_autocorrelation(factor_data, period=1)
```

These APIs are designed for repeated factor research workflows: compare
top-bottom spreads, inspect quantile stability, and monitor factor rank decay.
