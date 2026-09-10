# Quickstart

<script setup>
import { withBase } from 'vitepress'
</script>

Run the complete factor-analysis example after building from source:

```bash
python examples/factor_quickstart.py --output-dir quickstart-output
```

The example prepares a deterministic factor dataset, computes information
coefficient, quantile returns, factor returns, alpha/beta, turnover, and rank
autocorrelation, then writes:

- `quickstart-output/results.json`
- `quickstart-output/factor-report.html`

The Python package uses Polars as its default runtime dependency.

## Core Flow

```python
import ferric_alpha as fa

factor_data = fa.get_clean_factor_and_forward_returns(
    factor=factor,
    prices=prices,
    quantiles=5,
    periods=[1, 5, 10],
)

mean_ic = fa.performance.mean_information_coefficient(factor_data)
quantile_returns = fa.performance.mean_return_by_quantile(factor_data)
report = fa.tears.create_full_tear_sheet_data(factor_data)
fa.plotting.render(report).save("factor-report.html")
```

## Outputs

`results.json` is intended for automated checks and scripted research
workflows. `factor-report.html` is a self-contained native HTML tear sheet
that can be opened without Matplotlib.

## Visual Output

The quickstart also renders a native tear sheet. The preview below is generated
from the same deterministic example data used by `examples/factor_quickstart.py`.

<figure class="quickstart-preview">
  <img :src="withBase('/quickstart-report.svg')" alt="Ferric Alpha quickstart tear sheet preview">
</figure>

The full HTML report keeps the same charts and tables in a self-contained file:

```bash
open quickstart-output/factor-report.html
```
