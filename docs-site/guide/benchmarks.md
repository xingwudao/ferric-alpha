# Benchmarks

The benchmark below compares the Python-facing `ferric-alpha` API with
`alphalens` on the same deterministic long/short factor fixture. It measures
function execution time only; fixture construction is outside the timed block.

## Setup

- Machine: Apple silicon macOS `26.3.1`
- Dataset: `250` assets, `252` sessions, `63,000` factor rows
- Forward returns: `1D` and `5D`
- Iterations: `5`; result is median wall-clock time
- `ferric-alpha`: `0.1.0`, release build, Python `3.13.4`, Polars `1.42.1`
- `alphalens`: `0.4.0`, Python `3.9.25`, Pandas `1.5.3`, NumPy `1.23.5`

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
  - `ferric-alpha`: `16.541 ms`
  - `alphalens`: `14.279 ms`
  - Speedup: `0.9x`

Benchmark results are hardware and environment dependent.

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
