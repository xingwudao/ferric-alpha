# Benchmarks

<script setup>
import { withBase } from 'vitepress'
</script>

The benchmark below compares the Python-facing `ferric-alpha` API with
`alphalens` on the same deterministic long/short factor fixture. It measures
function execution time only; fixture construction is outside the timed block.

## Setup

- Machine: Apple silicon macOS `26.3.1`
- Dataset: `250` assets, `252` sessions, `63,000` factor rows
- Forward returns: `1D` and `5D`
- Iterations: `5`; result is median wall-clock time
- Rank autocorrelation recheck: median of `7` runs, `15` iterations per run
- `ferric-alpha`: `0.1.0`, release build, Python `3.13.4`, Polars `1.42.1`
- `alphalens`: `0.4.0`, Python `3.9.25`, Pandas `1.5.3`, NumPy `1.23.5`

<figure>
  <img :src="withBase('/performance-comparison.svg')" alt="Ferric Alpha performance comparison with alphalens">
</figure>

## Results

- `factor_information_coefficient`
  - `ferric-alpha`: `13.616 ms`
  - `alphalens`: `124.205 ms`
  - Speedup: `9.1x`
- `factor_weights`
  - `ferric-alpha`: `15.153 ms`
  - `alphalens`: `43.330 ms`
  - Speedup: `2.9x`
- `factor_returns`
  - `ferric-alpha`: `11.778 ms`
  - `alphalens`: `46.425 ms`
  - Speedup: `3.9x`
- `mean_return_by_quantile`
  - `ferric-alpha`: `21.022 ms`
  - `alphalens`: `60.900 ms`
  - Speedup: `2.9x`
- `quantile_turnover`
  - `ferric-alpha`: `3.018 ms`
  - `alphalens`: `6.136 ms`
  - Speedup: `2.0x`
- `factor_rank_autocorrelation`
  - `ferric-alpha`: `6.823 ms` median (`6.686-6.920 ms` observed)
  - `alphalens`: `15.497 ms` median (`15.326-15.864 ms` observed)
  - Speedup: `2.27x`

Benchmark results are hardware and environment dependent.

## Rank Autocorrelation Fast Path

Complete date-by-asset panels use a dense matrix internally. Assets are encoded
once, each date occupies one contiguous matrix row, and lagged rows are
correlated directly. This avoids per-date string maps while preserving the
public Python and Rust APIs.

Sparse panels, changing universes, null factors, and non-finite factors use the
general compatibility path. The optimization therefore changes execution, not
the metric definition.

The optimized path was checked directly against alphalens `0.4.0` using a
shuffled `80`-session by `37`-asset panel containing tied factor values. For
lags `1`, `3`, and `10`, null positions matched exactly and the maximum absolute
difference was `1.11e-16` under a `1e-12` comparison tolerance.

## Reproduce

```bash
.venv/bin/maturin develop --release
.venv/bin/python scripts/benchmark_alphalens_comparison.py --engine ferric

python3.9 -m venv /tmp/alphalens-bench
/tmp/alphalens-bench/bin/python -m pip install \
  'alphalens==0.4.0' 'pandas==1.5.3' 'numpy==1.23.5' \
  'scipy==1.10.1' 'statsmodels==0.13.5' 'empyrical==0.5.5'
PYTHONWARNINGS=ignore /tmp/alphalens-bench/bin/python \
  scripts/benchmark_alphalens_comparison.py --engine alphalens
```
