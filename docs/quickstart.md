# Factor Analysis Quickstart

This example runs a complete cross-sectional factor study with Polars and the
Ferric Alpha Python API. It requires no Pandas, SciPy, Statsmodels, or
AlphaLens installation.

## Install

Install the package from PyPI after the first release:

```bash
pip install ferric-alpha
```

For a source checkout, create the development environment and build the Rust
extension:

```bash
make develop
```

## Run The Complete Example

```bash
python examples/factor_quickstart.py --output-dir quickstart-output
```

The command writes two artifacts:

- `quickstart-output/results.json` contains key metrics and sanity checks.
- `quickstart-output/factor-report.html` is a self-contained native tear sheet.

The deterministic dataset has 20 assets, four industry groups, and 80 business
days. Its factor has a known positive relation to future returns, making the
example useful as both a tutorial and an installation smoke test.

## Input Data

Ferric Alpha uses long-form Polars DataFrames instead of a Pandas MultiIndex.
The factor input requires:

- `date`: Polars `Datetime`
- `asset`: Polars `String`
- `factor`: Polars `Float64`

The price input requires the same `date` and `asset` keys plus a positive
Float64 `price` column. Each `(date, asset)` key must be unique. An optional
String `group` column can be present in the factor frame, or groups can be
supplied as an asset-to-group mapping.

## Prepare Factor Data

The main preparation function computes forward returns, assigns quantiles,
joins groups, removes unusable rows, and reports the exact row loss:

```python
import ferric_alpha as fa

clean = fa.get_clean_factor_and_forward_returns(
    factor,
    prices,
    periods=(1, 5, 10),
    quantiles=5,
    groupby=group_by,
)

factor_data = clean.frame
print(clean.loss.input_rows, clean.loss.output_rows)
```

`factor_data` contains `date`, `asset`, `factor`, `group`,
`factor_quantile`, and columns such as `forward_return_1D`. The final rows for
each asset are removed when a complete requested forward horizon is not
available. Set `max_loss` explicitly when your research policy requires a
different threshold.

## Calculate Core Metrics

```python
performance = fa.performance

mean_ic = performance.mean_information_coefficient(factor_data)
quantile_returns = performance.mean_return_by_quantile(factor_data)
spread = performance.compute_mean_returns_spread(quantile_returns, 5, 1)
weights = performance.factor_weights(factor_data)
factor_returns = performance.factor_returns(factor_data)
alpha_beta = performance.factor_alpha_beta(factor_data)
turnover = performance.quantile_turnover(factor_data, quantile=5, period=1)
rank_autocorrelation = performance.factor_rank_autocorrelation(
    factor_data, period=1
)
```

For `factor_alpha_beta`, omitting `returns` uses the equal-weight universe as
the benchmark. A custom return series must contain `date`, `period`, and
`factor_return` columns with periods matching the clean factor data.

## Render A Tear Sheet

```python
report = fa.tears.create_full_tear_sheet_data(factor_data)
rendered = fa.plotting.render(report, backend="html", theme="light")
rendered.save("factor-report.html")
```

Native HTML, SVG, and PNG rendering use the Rust backend. HTML is suitable for
notebooks and produces a self-contained file. The optional Matplotlib backend
is available through `pip install 'ferric-alpha[plot]'`.

Replace the synthetic `factor` and `prices` frames in the example with your
own point-in-time data to use the same workflow in research. Factor timestamps
must not include information unavailable at the timestamp being evaluated.
