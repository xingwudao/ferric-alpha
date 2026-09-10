# Rendering

Native HTML is the default rendering path and does not require Matplotlib,
Pandas, Seaborn, SciPy, or Statsmodels.

```python
import ferric_alpha as fa

report = fa.tears.create_full_tear_sheet_data(factor_data)
rendered = fa.plotting.render(report)
rendered.save("factor-report.html")
```

## Native Formats

```python
svg = fa.plotting.render_svg(report)
png = fa.plotting.render_png(report)
html = fa.plotting.render_html(report)
```

The same native renderer powers notebook display:

```python
fa.plotting.display(report)
```

## Optional Matplotlib Backend

Install the plotting extra when you need a Matplotlib figure object:

```bash
pip install 'ferric-alpha[plot]'
```

```python
figure = fa.plotting.render(report, backend="matplotlib")
figure.savefig("factor-report.png", dpi=144)
```

The Matplotlib backend returns a `matplotlib.figure.Figure` and never calls
`show()` implicitly.
