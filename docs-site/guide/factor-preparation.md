# Factor Preparation

Factor preparation turns raw factor observations and prices into clean
long-form factor data with forward returns and quantile labels.

```python
import ferric_alpha as fa

factor_data = fa.get_clean_factor_and_forward_returns(
    factor=factor,
    prices=prices,
    quantiles=5,
    periods=[1, 5, 10],
    max_loss=0.35,
)
```

## Inputs

- `factor`: factor values indexed by date and asset-like keys.
- `prices`: price observations used to compute forward returns.
- `quantiles`: integer count or explicit quantile edges.
- `periods`: observed-session forward windows.

## Output Contract

The output is a Polars DataFrame with `date`, `asset`, `factor`,
`factor_quantile`, and one or more `forward_return_*` columns.

Ferric Alpha keeps missing terminal prices and invalid price windows explicit
as nulls instead of silently imputing forward returns.
