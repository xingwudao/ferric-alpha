#![cfg(feature = "extension-module")]

use ferric_alpha::FerricAlphaError;
use pyo3::exceptions::{PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use pyo3_polars::PyDataFrame;

#[pyclass(name = "LossReport", skip_from_py_object)]
#[derive(Clone)]
struct PyLossReport {
    #[pyo3(get)]
    input_rows: usize,
    #[pyo3(get)]
    output_rows: usize,
    #[pyo3(get)]
    forward_return_loss: usize,
    #[pyo3(get)]
    quantile_loss: usize,
}

impl From<ferric_alpha::LossReport> for PyLossReport {
    fn from(loss: ferric_alpha::LossReport) -> Self {
        Self {
            input_rows: loss.input_rows,
            output_rows: loss.output_rows,
            forward_return_loss: loss.forward_return_loss,
            quantile_loss: loss.quantile_loss,
        }
    }
}

#[pyclass(name = "CleanFactorResult")]
struct PyCleanFactorResult {
    frame: polars::prelude::DataFrame,
    loss: PyLossReport,
}

#[pyclass(name = "PyfolioInput")]
struct PyPyfolioInput {
    returns: polars::prelude::DataFrame,
    positions: polars::prelude::DataFrame,
    benchmark: Option<polars::prelude::DataFrame>,
}

#[pyclass(name = "TearSheetData", skip_from_py_object)]
#[derive(Clone)]
struct PyTearSheetData {
    report: ferric_alpha::TearSheetData,
}

#[pymethods]
impl PyPyfolioInput {
    #[getter]
    fn returns(&self) -> PyDataFrame {
        PyDataFrame(self.returns.clone())
    }

    #[getter]
    fn positions(&self) -> PyDataFrame {
        PyDataFrame(self.positions.clone())
    }

    #[getter]
    fn benchmark(&self) -> Option<PyDataFrame> {
        self.benchmark.clone().map(PyDataFrame)
    }
}

#[pymethods]
impl PyCleanFactorResult {
    #[getter]
    fn frame(&self) -> PyDataFrame {
        PyDataFrame(self.frame.clone())
    }

    #[getter]
    fn loss(&self) -> PyLossReport {
        self.loss.clone()
    }
}

#[pymethods]
impl PyTearSheetData {
    #[getter]
    fn kind(&self) -> &'static str {
        report_kind(&self.report)
    }

    #[getter]
    fn contract_version(&self) -> &str {
        report_metadata(&self.report).contract_version()
    }

    fn table_ids(&self) -> Vec<String> {
        self.report
            .table_ids()
            .into_iter()
            .map(ToOwned::to_owned)
            .collect()
    }

    fn table<'py>(&self, py: Python<'py>, id: &str) -> PyResult<Bound<'py, PyAny>> {
        let table = self
            .report
            .table(id)
            .ok_or_else(|| PyKeyError::new_err(id.to_owned()))?;
        let frame = table.to_polars().map_err(to_python_error)?;
        PyDataFrame(frame)
            .into_pyobject(py)
            .map_err(|error| PyRuntimeError::new_err(error.to_string()))
    }

    fn table_metadata<'py>(&self, py: Python<'py>, id: &str) -> PyResult<Bound<'py, PyDict>> {
        let table = self
            .report
            .table(id)
            .ok_or_else(|| PyKeyError::new_err(id.to_owned()))?;
        table_metadata(py, table)
    }

    #[pyo3(signature = (pretty=false))]
    fn to_json(&self, py: Python<'_>, pretty: bool) -> PyResult<String> {
        py.detach(|| {
            if pretty {
                self.report.to_pretty_json()
            } else {
                self.report.to_json()
            }
        })
        .map_err(to_python_error)
    }

    #[staticmethod]
    fn from_json(py: Python<'_>, value: &str) -> PyResult<Self> {
        py.detach(|| ferric_alpha::TearSheetData::from_json(value))
            .map(|report| Self { report })
            .map_err(to_python_error)
    }
}

#[pyfunction]
fn validate_factor_frame(py: Python<'_>, frame: PyDataFrame) -> PyResult<PyDataFrame> {
    let frame = frame.0;
    py.detach(|| ferric_alpha::validate_factor_frame(&frame))
        .map(PyDataFrame)
        .map_err(to_python_error)
}

#[pyfunction]
#[pyo3(signature = (factor, prices, periods, filter_zscore=None, cumulative_returns=true))]
fn compute_forward_returns(
    py: Python<'_>,
    factor: PyDataFrame,
    prices: PyDataFrame,
    periods: Vec<u32>,
    filter_zscore: Option<f64>,
    cumulative_returns: bool,
) -> PyResult<PyDataFrame> {
    let factor = factor.0;
    let prices = prices.0;
    py.detach(|| {
        let periods = to_periods(periods)?;
        ferric_alpha::compute_forward_returns_with_options(
            &factor,
            &prices,
            &periods,
            ferric_alpha::ForwardReturnsOptions {
                cumulative_returns,
                filter_zscore,
            },
        )
    })
    .map(PyDataFrame)
    .map_err(to_python_error)
}

#[pyfunction]
#[pyo3(signature = (factor, quantiles=None, bins=None, by_group=false, zero_aware=false))]
fn quantize_factor(
    py: Python<'_>,
    factor: PyDataFrame,
    quantiles: Option<Py<PyAny>>,
    bins: Option<Py<PyAny>>,
    by_group: bool,
    zero_aware: bool,
) -> PyResult<PyDataFrame> {
    let factor = factor.0;
    let options = quantize_options(py, quantiles, bins, by_group, zero_aware)?;
    py.detach(|| ferric_alpha::quantize_factor(&factor, options))
        .map(PyDataFrame)
        .map_err(to_python_error)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    factor,
    forward_returns,
    quantiles=None,
    bins=None,
    max_loss=0.35,
    by_group=false,
    zero_aware=false
))]
fn get_clean_factor(
    py: Python<'_>,
    factor: PyDataFrame,
    forward_returns: PyDataFrame,
    quantiles: Option<Py<PyAny>>,
    bins: Option<Py<PyAny>>,
    max_loss: f64,
    by_group: bool,
    zero_aware: bool,
) -> PyResult<PyCleanFactorResult> {
    let factor = factor.0;
    let forward_returns = forward_returns.0;
    let options = quantize_options(py, quantiles, bins, by_group, zero_aware)?;
    py.detach(|| {
        ferric_alpha::get_clean_factor(
            &factor,
            &forward_returns,
            options,
            ferric_alpha::CleanOptions { max_loss },
        )
    })
    .map(to_py_clean_result)
    .map_err(to_python_error)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    factor,
    prices,
    periods,
    quantiles=None,
    bins=None,
    max_loss=0.35,
    by_group=false,
    zero_aware=false,
    cumulative_returns=true,
    filter_zscore=None
))]
fn get_clean_factor_and_forward_returns(
    py: Python<'_>,
    factor: PyDataFrame,
    prices: PyDataFrame,
    periods: Vec<u32>,
    quantiles: Option<Py<PyAny>>,
    bins: Option<Py<PyAny>>,
    max_loss: f64,
    by_group: bool,
    zero_aware: bool,
    cumulative_returns: bool,
    filter_zscore: Option<f64>,
) -> PyResult<PyCleanFactorResult> {
    let factor = factor.0;
    let prices = prices.0;
    let options = quantize_options(py, quantiles, bins, by_group, zero_aware)?;
    py.detach(|| {
        let periods = to_periods(periods)?;
        ferric_alpha::get_clean_factor_and_forward_returns_with_options(
            &factor,
            &prices,
            &periods,
            options,
            ferric_alpha::CleanOptions { max_loss },
            ferric_alpha::ForwardReturnsOptions {
                cumulative_returns,
                filter_zscore,
            },
        )
    })
    .map(to_py_clean_result)
    .map_err(to_python_error)
}

fn to_periods(periods: Vec<u32>) -> ferric_alpha::Result<Vec<ferric_alpha::Period>> {
    periods.into_iter().map(ferric_alpha::Period::new).collect()
}

fn quantize_options(
    py: Python<'_>,
    quantiles: Option<Py<PyAny>>,
    bins: Option<Py<PyAny>>,
    by_group: bool,
    zero_aware: bool,
) -> PyResult<ferric_alpha::QuantizeOptions> {
    let quantiles = quantiles
        .as_ref()
        .map(|value| parse_quantiles(py, value))
        .transpose()?;
    let bins = bins
        .as_ref()
        .map(|value| parse_bins(py, value))
        .transpose()?;
    let options = match (quantiles, bins) {
        (Some(PyQuantiles::Count(quantiles)), None) => {
            ferric_alpha::QuantizeOptions::quantiles(quantiles)
        }
        (Some(PyQuantiles::Edges(edges)), None) => {
            ferric_alpha::QuantizeOptions::quantile_edges(edges)
        }
        (None, Some(PyBins::Count(bins))) => ferric_alpha::QuantizeOptions::bin_count(bins),
        (None, Some(PyBins::Edges(bins))) => ferric_alpha::QuantizeOptions::bins(bins),
        (None, None) => ferric_alpha::QuantizeOptions::quantiles(5),
        (Some(_), Some(_)) => {
            return Err(to_python_error(
                ferric_alpha::FerricAlphaError::ConflictingQuantileMode,
            ));
        }
    };
    Ok(options.with_by_group(by_group).with_zero_aware(zero_aware))
}

enum PyQuantiles {
    Count(u32),
    Edges(Vec<f64>),
}

enum PyBins {
    Count(u32),
    Edges(Vec<f64>),
}

fn parse_quantiles(py: Python<'_>, quantiles: &Py<PyAny>) -> PyResult<PyQuantiles> {
    let quantiles = quantiles.bind(py);
    if let Ok(count) = quantiles.extract::<u32>() {
        return Ok(PyQuantiles::Count(count));
    }
    if let Ok(edges) = quantiles.extract::<Vec<f64>>() {
        return Ok(PyQuantiles::Edges(edges));
    }
    Err(PyValueError::new_err(
        "quantiles must be an integer count or a probability edge list",
    ))
}

fn parse_bins(py: Python<'_>, bins: &Py<PyAny>) -> PyResult<PyBins> {
    let bins = bins.bind(py);
    if let Ok(count) = bins.extract::<u32>() {
        return Ok(PyBins::Count(count));
    }
    if let Ok(edges) = bins.extract::<Vec<f64>>() {
        return Ok(PyBins::Edges(edges));
    }
    Err(PyValueError::new_err(
        "bins must be an integer count or a numeric boundary list",
    ))
}

fn to_py_clean_result(result: ferric_alpha::CleanFactorResult) -> PyCleanFactorResult {
    PyCleanFactorResult {
        frame: result.frame,
        loss: result.loss.into(),
    }
}

#[pyfunction]
#[pyo3(signature = (factor_data, *, group_adjust=false, by_group=false))]
fn factor_information_coefficient(
    py: Python<'_>,
    factor_data: PyDataFrame,
    group_adjust: bool,
    by_group: bool,
) -> PyResult<PyDataFrame> {
    let factor_data = factor_data.0;
    py.detach(|| {
        ferric_alpha::factor_information_coefficient(
            &factor_data,
            ferric_alpha::IcOptions::default()
                .with_group_adjust(group_adjust)
                .with_by_group(by_group),
        )
    })
    .map(PyDataFrame)
    .map_err(to_python_error)
}

#[pyfunction]
#[pyo3(signature = (factor_data, *, group_adjust=false, by_group=false, by_time=None))]
fn mean_information_coefficient(
    py: Python<'_>,
    factor_data: PyDataFrame,
    group_adjust: bool,
    by_group: bool,
    by_time: Option<String>,
) -> PyResult<PyDataFrame> {
    let factor_data = factor_data.0;
    py.detach(|| {
        ferric_alpha::mean_information_coefficient(
            &factor_data,
            ferric_alpha::MeanIcOptions::default()
                .with_group_adjust(group_adjust)
                .with_by_group(by_group)
                .with_by_time(parse_time_bucket(by_time)?),
        )
    })
    .map(PyDataFrame)
    .map_err(to_python_error)
}

#[pyfunction]
#[pyo3(signature = (factor_data, *, demeaned=true, group_adjust=false, equal_weight=false))]
fn factor_weights(
    py: Python<'_>,
    factor_data: PyDataFrame,
    demeaned: bool,
    group_adjust: bool,
    equal_weight: bool,
) -> PyResult<PyDataFrame> {
    let factor_data = factor_data.0;
    py.detach(|| {
        ferric_alpha::factor_weights(
            &factor_data,
            weight_options(demeaned, group_adjust, equal_weight),
        )
    })
    .map(PyDataFrame)
    .map_err(to_python_error)
}

#[pyfunction]
#[pyo3(signature = (
    factor_data,
    *,
    demeaned=true,
    group_adjust=false,
    equal_weight=false,
    by_asset=false
))]
fn factor_returns(
    py: Python<'_>,
    factor_data: PyDataFrame,
    demeaned: bool,
    group_adjust: bool,
    equal_weight: bool,
    by_asset: bool,
) -> PyResult<PyDataFrame> {
    let factor_data = factor_data.0;
    py.detach(|| {
        ferric_alpha::factor_returns(
            &factor_data,
            ferric_alpha::ReturnOptions::default()
                .with_weight_options(weight_options(demeaned, group_adjust, equal_weight))
                .with_by_asset(by_asset),
        )
    })
    .map(PyDataFrame)
    .map_err(to_python_error)
}

#[pyfunction]
#[pyo3(signature = (
    factor_data,
    returns=None,
    *,
    demeaned=true,
    group_adjust=false,
    equal_weight=false
))]
fn factor_alpha_beta(
    py: Python<'_>,
    factor_data: PyDataFrame,
    returns: Option<PyDataFrame>,
    demeaned: bool,
    group_adjust: bool,
    equal_weight: bool,
) -> PyResult<PyDataFrame> {
    let factor_data = factor_data.0;
    let returns = returns.map(|returns| returns.0);
    py.detach(|| {
        ferric_alpha::factor_alpha_beta(
            &factor_data,
            returns.as_ref(),
            ferric_alpha::AlphaBetaOptions::default().with_weight_options(weight_options(
                demeaned,
                group_adjust,
                equal_weight,
            )),
        )
    })
    .map(PyDataFrame)
    .map_err(to_python_error)
}

#[pyfunction]
#[pyo3(signature = (
    factor_data,
    *,
    by_date=false,
    by_group=false,
    demeaned=true,
    group_adjust=false
))]
fn mean_return_by_quantile(
    py: Python<'_>,
    factor_data: PyDataFrame,
    by_date: bool,
    by_group: bool,
    demeaned: bool,
    group_adjust: bool,
) -> PyResult<PyDataFrame> {
    let factor_data = factor_data.0;
    py.detach(|| {
        ferric_alpha::mean_return_by_quantile(
            &factor_data,
            ferric_alpha::MeanReturnByQuantileOptions::default()
                .with_by_date(by_date)
                .with_by_group(by_group)
                .with_demeaned(demeaned)
                .with_group_adjust(group_adjust),
        )
    })
    .map(PyDataFrame)
    .map_err(to_python_error)
}

#[pyfunction]
#[pyo3(signature = (mean_returns, upper_quant, lower_quant, std_err=None))]
fn compute_mean_returns_spread(
    py: Python<'_>,
    mean_returns: PyDataFrame,
    upper_quant: u32,
    lower_quant: u32,
    std_err: Option<PyDataFrame>,
) -> PyResult<PyDataFrame> {
    let mean_returns = mean_returns.0;
    let std_err = std_err.map(|frame| frame.0);
    py.detach(|| {
        ferric_alpha::compute_mean_returns_spread(
            &mean_returns,
            upper_quant,
            lower_quant,
            std_err.as_ref(),
        )
    })
    .map(PyDataFrame)
    .map_err(to_python_error)
}

#[pyfunction]
#[pyo3(signature = (quantile_factor, quantile, period=1))]
fn quantile_turnover(
    py: Python<'_>,
    quantile_factor: PyDataFrame,
    quantile: i64,
    period: i64,
) -> PyResult<PyDataFrame> {
    let quantile_factor = quantile_factor.0;
    py.detach(|| {
        ferric_alpha::quantile_turnover(
            &quantile_factor,
            non_negative_u32(quantile, "quantile")?,
            phase_03_period(period)?,
        )
    })
    .map(PyDataFrame)
    .map_err(to_python_error)
}

#[pyfunction]
#[pyo3(signature = (factor_data, period=1))]
fn factor_rank_autocorrelation(
    py: Python<'_>,
    factor_data: PyDataFrame,
    period: i64,
) -> PyResult<PyDataFrame> {
    let factor_data = factor_data.0;
    py.detach(|| ferric_alpha::factor_rank_autocorrelation(&factor_data, phase_03_period(period)?))
        .map(PyDataFrame)
        .map_err(to_python_error)
}

#[pyfunction]
fn cumulative_returns(py: Python<'_>, returns: PyDataFrame) -> PyResult<PyDataFrame> {
    let returns = returns.0;
    py.detach(|| ferric_alpha::cumulative_returns(&returns))
        .map(PyDataFrame)
        .map_err(to_python_error)
}

#[pyfunction]
#[pyo3(signature = (weights, period, sessions=None))]
fn positions(
    py: Python<'_>,
    weights: PyDataFrame,
    period: i64,
    sessions: Option<PyDataFrame>,
) -> PyResult<PyDataFrame> {
    let weights = weights.0;
    let sessions = sessions.map(|sessions| sessions.0);
    py.detach(|| ferric_alpha::positions(&weights, phase_03_period(period)?, sessions.as_ref()))
        .map(PyDataFrame)
        .map_err(to_python_error)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    factor_data,
    period,
    long_short=true,
    group_neutral=false,
    equal_weight=false,
    quantiles=None,
    groups=None
))]
fn factor_cumulative_returns(
    py: Python<'_>,
    factor_data: PyDataFrame,
    period: i64,
    long_short: bool,
    group_neutral: bool,
    equal_weight: bool,
    quantiles: Option<Vec<i64>>,
    groups: Option<Vec<String>>,
) -> PyResult<PyDataFrame> {
    let factor_data = factor_data.0;
    py.detach(|| {
        ferric_alpha::factor_cumulative_returns(
            &factor_data,
            phase_03_period(period)?,
            portfolio_options(long_short, group_neutral, equal_weight, quantiles, groups)?,
        )
    })
    .map(PyDataFrame)
    .map_err(to_python_error)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    factor_data,
    period,
    long_short=true,
    group_neutral=false,
    equal_weight=false,
    quantiles=None,
    groups=None,
    sessions=None
))]
fn factor_positions(
    py: Python<'_>,
    factor_data: PyDataFrame,
    period: i64,
    long_short: bool,
    group_neutral: bool,
    equal_weight: bool,
    quantiles: Option<Vec<i64>>,
    groups: Option<Vec<String>>,
    sessions: Option<PyDataFrame>,
) -> PyResult<PyDataFrame> {
    let factor_data = factor_data.0;
    let sessions = sessions.map(|sessions| sessions.0);
    py.detach(|| {
        ferric_alpha::factor_positions(
            &factor_data,
            phase_03_period(period)?,
            portfolio_options(long_short, group_neutral, equal_weight, quantiles, groups)?,
            sessions.as_ref(),
        )
    })
    .map(PyDataFrame)
    .map_err(to_python_error)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    factor_data,
    period,
    capital=None,
    long_short=true,
    group_neutral=false,
    equal_weight=false,
    quantiles=None,
    groups=None,
    benchmark_period=Some(1),
    sessions=None
))]
fn create_pyfolio_input(
    py: Python<'_>,
    factor_data: PyDataFrame,
    period: i64,
    capital: Option<f64>,
    long_short: bool,
    group_neutral: bool,
    equal_weight: bool,
    quantiles: Option<Vec<i64>>,
    groups: Option<Vec<String>>,
    benchmark_period: Option<i64>,
    sessions: Option<PyDataFrame>,
) -> PyResult<PyPyfolioInput> {
    let factor_data = factor_data.0;
    let sessions = sessions.map(|sessions| sessions.0);
    py.detach(|| {
        ferric_alpha::create_pyfolio_input(
            &factor_data,
            phase_03_period(period)?,
            capital,
            portfolio_options(long_short, group_neutral, equal_weight, quantiles, groups)?,
            benchmark_period.map(phase_03_period).transpose()?,
            sessions.as_ref(),
        )
    })
    .map(|result| PyPyfolioInput {
        returns: result.returns,
        positions: result.positions,
        benchmark: result.benchmark,
    })
    .map_err(to_python_error)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    factor_data,
    returns,
    periods_before=10,
    periods_after=15,
    demeaned=true,
    group_adjust=false,
    by_group=false
))]
fn average_cumulative_return_by_quantile(
    py: Python<'_>,
    factor_data: PyDataFrame,
    returns: PyDataFrame,
    periods_before: u32,
    periods_after: u32,
    demeaned: bool,
    group_adjust: bool,
    by_group: bool,
) -> PyResult<PyDataFrame> {
    let factor_data = factor_data.0;
    let returns = returns.0;
    py.detach(|| {
        ferric_alpha::average_cumulative_return_by_quantile(
            &factor_data,
            &returns,
            ferric_alpha::EventReturnsOptions {
                periods_before,
                periods_after,
                demeaned,
                group_adjust,
                by_group,
            },
        )
    })
    .map(PyDataFrame)
    .map_err(to_python_error)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    factor_data,
    returns,
    periods_before,
    periods_after,
    cumulative=false,
    mean_by_date=false,
    demeaned=false
))]
fn common_start_returns(
    py: Python<'_>,
    factor_data: PyDataFrame,
    returns: PyDataFrame,
    periods_before: u32,
    periods_after: u32,
    cumulative: bool,
    mean_by_date: bool,
    demeaned: bool,
) -> PyResult<PyDataFrame> {
    let factor_data = factor_data.0;
    let returns = returns.0;
    py.detach(|| {
        ferric_alpha::common_start_returns(
            &factor_data,
            &returns,
            ferric_alpha::CommonStartReturnsOptions::default()
                .with_periods_before(periods_before)
                .with_periods_after(periods_after)
                .with_cumulative(cumulative)
                .with_mean_by_date(mean_by_date)
                .with_demeaned(demeaned),
        )
    })
    .map(PyDataFrame)
    .map_err(to_python_error)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    factor_data,
    returns,
    periods_before=10,
    periods_after=15,
    demeaned=true,
    group_adjust=false,
    by_group=false
))]
fn average_cumulative_return_by_quantile_from_prices(
    py: Python<'_>,
    factor_data: PyDataFrame,
    returns: PyDataFrame,
    periods_before: u32,
    periods_after: u32,
    demeaned: bool,
    group_adjust: bool,
    by_group: bool,
) -> PyResult<PyDataFrame> {
    let factor_data = factor_data.0;
    let returns = returns.0;
    py.detach(|| {
        ferric_alpha::average_cumulative_return_by_quantile_from_prices(
            &factor_data,
            &returns,
            ferric_alpha::EventReturnsOptions {
                periods_before,
                periods_after,
                demeaned,
                group_adjust,
                by_group,
            },
        )
    })
    .map(PyDataFrame)
    .map_err(to_python_error)
}

#[pyfunction]
#[pyo3(signature = (factor_data, long_short=true, group_neutral=false, turnover_periods=None))]
fn create_summary_tear_sheet_data(
    py: Python<'_>,
    factor_data: PyDataFrame,
    long_short: bool,
    group_neutral: bool,
    turnover_periods: Option<Vec<u32>>,
) -> PyResult<PyTearSheetData> {
    let factor_data = factor_data.0;
    py.detach(|| {
        ferric_alpha::create_summary_tear_sheet_data(
            &factor_data,
            ferric_alpha::SummaryTearSheetOptions {
                long_short,
                group_neutral,
                turnover_periods: turnover_periods.map(to_periods).transpose()?,
            },
        )
    })
    .map(|report| PyTearSheetData {
        report: ferric_alpha::TearSheetData::from(report),
    })
    .map_err(to_python_error)
}

#[pyfunction]
#[pyo3(signature = (factor_data, long_short=true, group_neutral=false, by_group=false))]
fn create_returns_tear_sheet_data(
    py: Python<'_>,
    factor_data: PyDataFrame,
    long_short: bool,
    group_neutral: bool,
    by_group: bool,
) -> PyResult<PyTearSheetData> {
    let factor_data = factor_data.0;
    py.detach(|| {
        ferric_alpha::create_returns_tear_sheet_data(
            &factor_data,
            ferric_alpha::ReturnsTearSheetOptions {
                long_short,
                group_neutral,
                by_group,
            },
        )
    })
    .map(|report| PyTearSheetData {
        report: ferric_alpha::TearSheetData::from(report),
    })
    .map_err(to_python_error)
}

#[pyfunction]
#[pyo3(signature = (factor_data, group_neutral=false, by_group=false, rolling_window=22))]
fn create_information_tear_sheet_data(
    py: Python<'_>,
    factor_data: PyDataFrame,
    group_neutral: bool,
    by_group: bool,
    rolling_window: u32,
) -> PyResult<PyTearSheetData> {
    let factor_data = factor_data.0;
    py.detach(|| {
        ferric_alpha::create_information_tear_sheet_data(
            &factor_data,
            ferric_alpha::InformationTearSheetOptions {
                group_neutral,
                by_group,
                rolling_window,
            },
        )
    })
    .map(|report| PyTearSheetData {
        report: ferric_alpha::TearSheetData::from(report),
    })
    .map_err(to_python_error)
}

#[pyfunction]
#[pyo3(signature = (factor_data, turnover_periods=None))]
fn create_turnover_tear_sheet_data(
    py: Python<'_>,
    factor_data: PyDataFrame,
    turnover_periods: Option<Vec<u32>>,
) -> PyResult<PyTearSheetData> {
    let factor_data = factor_data.0;
    py.detach(|| {
        ferric_alpha::create_turnover_tear_sheet_data(
            &factor_data,
            ferric_alpha::TurnoverTearSheetOptions {
                periods: turnover_periods.map(to_periods).transpose()?,
            },
        )
    })
    .map(|report| PyTearSheetData {
        report: ferric_alpha::TearSheetData::from(report),
    })
    .map_err(to_python_error)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    factor_data,
    long_short=true,
    group_neutral=false,
    by_group=false,
    turnover_periods=None,
    ic_rolling_window=22
))]
fn create_full_tear_sheet_data(
    py: Python<'_>,
    factor_data: PyDataFrame,
    long_short: bool,
    group_neutral: bool,
    by_group: bool,
    turnover_periods: Option<Vec<u32>>,
    ic_rolling_window: u32,
) -> PyResult<PyTearSheetData> {
    let factor_data = factor_data.0;
    py.detach(|| {
        ferric_alpha::create_full_tear_sheet_data(
            &factor_data,
            ferric_alpha::FullTearSheetOptions {
                long_short,
                group_neutral,
                by_group,
                turnover_periods: turnover_periods.map(to_periods).transpose()?,
                ic_rolling_window,
            },
        )
    })
    .map(|report| PyTearSheetData {
        report: ferric_alpha::TearSheetData::from(report),
    })
    .map_err(to_python_error)
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
#[pyo3(signature = (
    factor_data,
    returns,
    periods_before=5,
    periods_after=15,
    long_short=true,
    group_neutral=false,
    by_group=false
))]
fn create_event_returns_tear_sheet_data(
    py: Python<'_>,
    factor_data: PyDataFrame,
    returns: PyDataFrame,
    periods_before: u32,
    periods_after: u32,
    long_short: bool,
    group_neutral: bool,
    by_group: bool,
) -> PyResult<PyTearSheetData> {
    let factor_data = factor_data.0;
    let returns = returns.0;
    py.detach(|| {
        ferric_alpha::create_event_returns_tear_sheet_data(
            &factor_data,
            &returns,
            ferric_alpha::EventReturnsOptions {
                periods_before,
                periods_after,
                demeaned: long_short,
                group_adjust: group_neutral,
                by_group,
            },
        )
    })
    .map(|report| PyTearSheetData {
        report: ferric_alpha::TearSheetData::from(report),
    })
    .map_err(to_python_error)
}

#[pyfunction]
#[pyo3(signature = (factor_data, returns=None, avgretplot=None, rate_of_ret=true, n_bars=50))]
fn create_event_study_tear_sheet_data(
    py: Python<'_>,
    factor_data: PyDataFrame,
    returns: Option<PyDataFrame>,
    avgretplot: Option<Vec<u32>>,
    rate_of_ret: bool,
    n_bars: u32,
) -> PyResult<PyTearSheetData> {
    let factor_data = factor_data.0;
    let returns = returns.map(|returns| returns.0);
    let event_window = match avgretplot {
        Some(values) if values.len() == 2 => Some((values[0], values[1])),
        Some(_) => {
            return Err(PyValueError::new_err(
                "invalid option avgretplot: expected a two-value window",
            ));
        }
        None => None,
    };
    py.detach(|| {
        ferric_alpha::create_event_study_tear_sheet_data(
            &factor_data,
            returns.as_ref(),
            ferric_alpha::EventStudyTearSheetOptions {
                event_window,
                rate_of_return: rate_of_ret,
                histogram_bins: n_bars,
            },
        )
    })
    .map(|report| PyTearSheetData {
        report: ferric_alpha::TearSheetData::from(report),
    })
    .map_err(to_python_error)
}

#[pyfunction]
#[pyo3(signature = (report, width=1200, scale=1.0, theme="light"))]
fn _plan_tear_sheet(
    py: Python<'_>,
    report: &PyTearSheetData,
    width: u32,
    scale: f64,
    theme: &str,
) -> PyResult<String> {
    let options = render_options(width, scale, theme)?;
    py.detach(|| ferric_alpha_render::plan_report(&report.report, &options)?.to_json())
        .map_err(to_python_render_error)
}

#[pyfunction]
#[pyo3(signature = (report, width=1200, scale=1.0, theme="light"))]
fn _render_tear_sheet_svg(
    py: Python<'_>,
    report: &PyTearSheetData,
    width: u32,
    scale: f64,
    theme: &str,
) -> PyResult<(String, u32, u32)> {
    let options = render_options(width, scale, theme)?;
    py.detach(|| {
        let plan = ferric_alpha_render::plan_report(&report.report, &options)?;
        let svg = ferric_alpha_render::render_svg(&report.report, &options)?;
        Ok::<_, ferric_alpha_render::RenderError>((svg, plan.width, plan.height))
    })
    .map_err(to_python_render_error)
}

#[pyfunction]
#[pyo3(signature = (report, width=1200, scale=1.0, theme="light"))]
fn _render_tear_sheet_png(
    py: Python<'_>,
    report: &PyTearSheetData,
    width: u32,
    scale: f64,
    theme: &str,
) -> PyResult<(Py<PyBytes>, u32, u32)> {
    let options = render_options(width, scale, theme)?;
    let physical = py
        .detach(|| {
            let plan = ferric_alpha_render::plan_report(&report.report, &options)?;
            let physical = options.validate(plan.height)?;
            let png = ferric_alpha_render::render_png(&report.report, &options)?;
            Ok::<_, ferric_alpha_render::RenderError>((png, physical.width, physical.height))
        })
        .map_err(to_python_render_error)?;
    Ok((
        PyBytes::new(py, &physical.0).unbind(),
        physical.1,
        physical.2,
    ))
}

#[pyfunction]
#[pyo3(signature = (report, width=1200, scale=1.0, theme="light"))]
fn _render_tear_sheet_html(
    py: Python<'_>,
    report: &PyTearSheetData,
    width: u32,
    scale: f64,
    theme: &str,
) -> PyResult<(String, u32, u32)> {
    let options = render_options(width, scale, theme)?;
    py.detach(|| {
        let plan = ferric_alpha_render::plan_report(&report.report, &options)?;
        let html = ferric_alpha_render::render_html_fragment(&report.report, &options)?;
        Ok::<_, ferric_alpha_render::RenderError>((html, plan.width, plan.height))
    })
    .map_err(to_python_render_error)
}

fn weight_options(
    demeaned: bool,
    group_adjust: bool,
    equal_weight: bool,
) -> ferric_alpha::WeightOptions {
    ferric_alpha::WeightOptions::default()
        .with_demeaned(demeaned)
        .with_group_adjust(group_adjust)
        .with_equal_weight(equal_weight)
}

fn portfolio_options(
    long_short: bool,
    group_neutral: bool,
    equal_weight: bool,
    quantiles: Option<Vec<i64>>,
    groups: Option<Vec<String>>,
) -> ferric_alpha::Result<ferric_alpha::PortfolioOptions> {
    let quantiles = quantiles
        .map(|values| {
            values
                .into_iter()
                .map(|value| non_negative_u32(value, "quantiles"))
                .collect()
        })
        .transpose()?;
    Ok(ferric_alpha::PortfolioOptions::default()
        .with_weight_options(weight_options(long_short, group_neutral, equal_weight))
        .with_quantiles(quantiles)
        .with_groups(groups))
}

fn phase_03_period(value: i64) -> ferric_alpha::Result<ferric_alpha::Period> {
    ferric_alpha::Period::new(non_negative_u32(value, "period")?)
}

fn non_negative_u32(value: i64, option: &'static str) -> ferric_alpha::Result<u32> {
    u32::try_from(value).map_err(|_| ferric_alpha::FerricAlphaError::InvalidOption {
        option,
        reason: "value must fit a non-negative 32-bit integer",
    })
}

fn parse_time_bucket(
    value: Option<String>,
) -> ferric_alpha::Result<Option<ferric_alpha::TimeBucket>> {
    match value.as_deref() {
        None => Ok(None),
        Some("daily") => Ok(Some(ferric_alpha::TimeBucket::Daily)),
        Some("weekly") => Ok(Some(ferric_alpha::TimeBucket::Weekly)),
        Some("monthly") => Ok(Some(ferric_alpha::TimeBucket::Monthly)),
        Some(_) => Err(ferric_alpha::FerricAlphaError::InvalidOption {
            option: "by_time",
            reason: "expected None, daily, weekly, or monthly",
        }),
    }
}

fn render_options(
    width: u32,
    scale: f64,
    theme: &str,
) -> PyResult<ferric_alpha_render::RenderOptions> {
    let theme = match theme {
        "light" => ferric_alpha_render::RenderTheme::Light,
        "dark" => ferric_alpha_render::RenderTheme::Dark,
        _ => return Err(PyValueError::new_err("invalid option theme")),
    };
    Ok(ferric_alpha_render::RenderOptions {
        width,
        scale,
        theme,
    })
}

fn to_python_render_error(error: ferric_alpha_render::RenderError) -> PyErr {
    match error {
        ferric_alpha_render::RenderError::InvalidOption { .. }
        | ferric_alpha_render::RenderError::MissingTable { .. }
        | ferric_alpha_render::RenderError::MissingColumn { .. }
        | ferric_alpha_render::RenderError::InvalidColumnType { .. }
        | ferric_alpha_render::RenderError::InvalidContract { .. }
        | ferric_alpha_render::RenderError::LayoutOverflow => {
            PyValueError::new_err(error.to_string())
        }
        ferric_alpha_render::RenderError::Allocation { .. }
        | ferric_alpha_render::RenderError::OutputTooLarge { .. }
        | ferric_alpha_render::RenderError::Backend { .. }
        | ferric_alpha_render::RenderError::PngEncoding { .. }
        | ferric_alpha_render::RenderError::Io { .. } => PyRuntimeError::new_err(error.to_string()),
    }
}

fn to_python_error(error: FerricAlphaError) -> PyErr {
    if matches!(&error, FerricAlphaError::Polars(_)) {
        PyRuntimeError::new_err(error.to_string())
    } else {
        PyValueError::new_err(error.to_string())
    }
}

fn report_metadata(report: &ferric_alpha::TearSheetData) -> &ferric_alpha::ReportMetadata {
    match report {
        ferric_alpha::TearSheetData::Summary(report) => &report.metadata,
        ferric_alpha::TearSheetData::Returns(report) => &report.metadata,
        ferric_alpha::TearSheetData::Information(report) => &report.metadata,
        ferric_alpha::TearSheetData::Turnover(report) => &report.metadata,
        ferric_alpha::TearSheetData::Full(report) => &report.metadata,
        ferric_alpha::TearSheetData::EventReturns(report) => &report.metadata,
        ferric_alpha::TearSheetData::EventStudy(report) => &report.metadata,
    }
}

fn report_kind(report: &ferric_alpha::TearSheetData) -> &'static str {
    match report {
        ferric_alpha::TearSheetData::Summary(_) => "summary",
        ferric_alpha::TearSheetData::Returns(_) => "returns",
        ferric_alpha::TearSheetData::Information(_) => "information",
        ferric_alpha::TearSheetData::Turnover(_) => "turnover",
        ferric_alpha::TearSheetData::Full(_) => "full",
        ferric_alpha::TearSheetData::EventReturns(_) => "event_returns",
        ferric_alpha::TearSheetData::EventStudy(_) => "event_study",
    }
}

fn table_metadata<'py>(
    py: Python<'py>,
    table: &ferric_alpha::ReportTable,
) -> PyResult<Bound<'py, PyDict>> {
    let metadata = PyDict::new(py);
    metadata.set_item("id", table.id())?;
    metadata.set_item("title", table.title())?;
    metadata.set_item("row_count", table.row_count())?;
    metadata.set_item("sort_by", PyList::new(py, table.sort_by().to_vec())?)?;

    let columns = PyList::empty(py);
    for column in table.columns() {
        let column_metadata = PyDict::new(py);
        column_metadata.set_item("name", column.name())?;
        column_metadata.set_item("label", column.label())?;
        column_metadata.set_item("role", column_role(column.role()))?;
        column_metadata.set_item("dtype", column_data_type(column.data()))?;
        if let Some(display) = column.display() {
            let display_metadata = PyDict::new(py);
            display_metadata.set_item("unit", display_unit(display.unit()))?;
            display_metadata.set_item("decimals", display.decimals())?;
            column_metadata.set_item("display", display_metadata)?;
        } else {
            column_metadata.set_item("display", py.None())?;
        }
        columns.append(column_metadata)?;
    }
    metadata.set_item("columns", columns)?;
    Ok(metadata)
}

fn column_role(role: ferric_alpha::ColumnRole) -> &'static str {
    match role {
        ferric_alpha::ColumnRole::Dimension => "dimension",
        ferric_alpha::ColumnRole::Measure => "measure",
        ferric_alpha::ColumnRole::Statistic => "statistic",
    }
}

fn display_unit(unit: ferric_alpha::DisplayUnit) -> &'static str {
    match unit {
        ferric_alpha::DisplayUnit::Raw => "raw",
        ferric_alpha::DisplayUnit::DecimalReturn => "decimal_return",
        ferric_alpha::DisplayUnit::BasisPoints => "basis_points",
        ferric_alpha::DisplayUnit::Percent => "percent",
        ferric_alpha::DisplayUnit::Ratio => "ratio",
        ferric_alpha::DisplayUnit::Count => "count",
    }
}

fn column_data_type(data: &ferric_alpha::ReportColumnData) -> &'static str {
    match data {
        ferric_alpha::ReportColumnData::Boolean(_) => "boolean",
        ferric_alpha::ReportColumnData::Int32(_) => "int32",
        ferric_alpha::ReportColumnData::Int64(_) => "int64",
        ferric_alpha::ReportColumnData::UInt32(_) => "uint32",
        ferric_alpha::ReportColumnData::UInt64(_) => "uint64",
        ferric_alpha::ReportColumnData::Float64(_) => "float64",
        ferric_alpha::ReportColumnData::String(_) => "string",
        ferric_alpha::ReportColumnData::Datetime { .. } => "datetime",
    }
}

#[pymodule]
fn _ferric_alpha(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyLossReport>()?;
    module.add_class::<PyCleanFactorResult>()?;
    module.add_class::<PyPyfolioInput>()?;
    module.add_class::<PyTearSheetData>()?;
    module.add_function(wrap_pyfunction!(validate_factor_frame, module)?)?;
    module.add_function(wrap_pyfunction!(compute_forward_returns, module)?)?;
    module.add_function(wrap_pyfunction!(quantize_factor, module)?)?;
    module.add_function(wrap_pyfunction!(get_clean_factor, module)?)?;
    module.add_function(wrap_pyfunction!(
        get_clean_factor_and_forward_returns,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(factor_information_coefficient, module)?)?;
    module.add_function(wrap_pyfunction!(mean_information_coefficient, module)?)?;
    module.add_function(wrap_pyfunction!(factor_weights, module)?)?;
    module.add_function(wrap_pyfunction!(factor_returns, module)?)?;
    module.add_function(wrap_pyfunction!(factor_alpha_beta, module)?)?;
    module.add_function(wrap_pyfunction!(mean_return_by_quantile, module)?)?;
    module.add_function(wrap_pyfunction!(compute_mean_returns_spread, module)?)?;
    module.add_function(wrap_pyfunction!(quantile_turnover, module)?)?;
    module.add_function(wrap_pyfunction!(factor_rank_autocorrelation, module)?)?;
    module.add_function(wrap_pyfunction!(cumulative_returns, module)?)?;
    module.add_function(wrap_pyfunction!(positions, module)?)?;
    module.add_function(wrap_pyfunction!(factor_cumulative_returns, module)?)?;
    module.add_function(wrap_pyfunction!(factor_positions, module)?)?;
    module.add_function(wrap_pyfunction!(create_pyfolio_input, module)?)?;
    module.add_function(wrap_pyfunction!(
        average_cumulative_return_by_quantile,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(common_start_returns, module)?)?;
    module.add_function(wrap_pyfunction!(
        average_cumulative_return_by_quantile_from_prices,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(create_summary_tear_sheet_data, module)?)?;
    module.add_function(wrap_pyfunction!(create_returns_tear_sheet_data, module)?)?;
    module.add_function(wrap_pyfunction!(
        create_information_tear_sheet_data,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(create_turnover_tear_sheet_data, module)?)?;
    module.add_function(wrap_pyfunction!(create_full_tear_sheet_data, module)?)?;
    module.add_function(wrap_pyfunction!(
        create_event_returns_tear_sheet_data,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(
        create_event_study_tear_sheet_data,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(_plan_tear_sheet, module)?)?;
    module.add_function(wrap_pyfunction!(_render_tear_sheet_svg, module)?)?;
    module.add_function(wrap_pyfunction!(_render_tear_sheet_png, module)?)?;
    module.add_function(wrap_pyfunction!(_render_tear_sheet_html, module)?)?;
    Ok(())
}
