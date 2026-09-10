# Data Model

Ferric Alpha uses explicit long-form columns:

- `date`: Polars `Datetime`
- `asset`: Polars `String`
- `factor`: Polars `Float64`
- `group`: optional Polars `String`
- `factor_quantile`: unsigned integer quantile label

Forward-return columns use names such as `forward_return_1D` and
`forward_return_5D`.

Ferric Alpha does not emulate Pandas `MultiIndex` behavior.

## Minimal Factor Frame

```python
import polars as pl
import ferric_alpha as fa

factor_data = pl.DataFrame(
    {
        "date": [...],
        "asset": [...],
        "factor": [...],
        "factor_quantile": [...],
        "forward_return_1D": [...],
    }
)

validated = fa.validate_factor_frame(factor_data)
```

The `(date, asset)` key must be non-null and unique. Null factor values are
preserved so downstream loss accounting can remain explicit.
