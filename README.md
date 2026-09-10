# Ferric Alpha

Ferric Alpha is a high-performance Rust implementation of factor analysis
workflows inspired by Alphalens. It uses Polars as its core table engine and
provides both Rust and Python APIs.

The project is under active development. No public release is available yet.

## Current scope

Ferric Alpha currently supports factor data preparation, core performance
metrics, turnover and portfolio series, serializable tear-sheet data, and
native report rendering.

## Data model

Core APIs use explicit long-form columns such as `date`, `asset`, and `factor`.
They do not emulate Pandas `MultiIndex` behavior.

Required columns:

- `date`: Polars `Datetime`
- `asset`: Polars `String`
- `factor`: Polars `Float64`; null values are preserved

The optional `group` column must be Polars `String`. The `(date, asset)` key
must be non-null and unique.

## Development

Prerequisites are Python 3.10 or newer, Rust 1.91 or newer, and `make`.

Create the local environment and build the extension:

```bash
make develop
```

Run all formatting checks, linters, Rust tests, and Python tests:

```bash
make verify
```

The Python package has Polars as its only default runtime dependency.

```python
import ferric_alpha

validated = ferric_alpha.validate_factor_frame(factor_frame)
```

## Rendering

Native HTML is the default plotting path and does not require Matplotlib,
Pandas, Seaborn, SciPy, or Statsmodels.

```python
import ferric_alpha as fa

report = fa.tears.create_full_tear_sheet_data(factor_data)
rendered = fa.plotting.render(report)
rendered.save("factor-report.html")
```

Native SVG and PNG are available directly:

```python
svg = fa.plotting.render_svg(report)
png = fa.plotting.render_png(report)
```

Notebook display uses the same native HTML object:

```python
fa.plotting.display(report)
```

The optional Matplotlib backend is installed separately:

```bash
pip install 'ferric-alpha[plot]'
```

```python
figure = fa.plotting.render(report, backend="matplotlib")
figure.savefig("factor-report.png", dpi=144)
```

The Matplotlib backend returns a `matplotlib.figure.Figure` and never calls
`show()` implicitly.

## License

Ferric Alpha is available under the Zero-Clause BSD (`0BSD`) license.
