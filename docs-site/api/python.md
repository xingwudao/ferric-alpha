# Python API

Ferric Alpha exposes Python modules for data preparation, performance metrics,
tear-sheet construction, rendering, and utility helpers.

## Modules

- `ferric_alpha`: package entry point and data preparation helpers.
- `ferric_alpha.performance`: factor metrics, returns, turnover, and portfolio
  series.
- `ferric_alpha.tears`: serializable tear-sheet data construction.
- `ferric_alpha.plotting`: native and optional Matplotlib rendering.
- `ferric_alpha.utils`: compatibility helpers and shared utility functions.

## Example

```python
import ferric_alpha as fa

report = fa.tears.create_full_tear_sheet_data(factor_data)
rendered = fa.plotting.render(report)
rendered.save("factor-report.html")
```

## Default Dependencies

The Python package uses Polars as its only default runtime dependency. Plotting
through Matplotlib is optional and installed with the `plot` extra.
