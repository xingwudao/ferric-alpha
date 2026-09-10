#![forbid(unsafe_code)]

mod calendar;
mod clean;
mod data;
mod error;
mod forward_returns;
mod performance;
mod quantiles;
mod report;
mod statistics;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub use calendar::{Period, TradingCalendar};
pub use clean::{
    CleanFactorResult, CleanOptions, LossReport, get_clean_factor,
    get_clean_factor_and_forward_returns, get_clean_factor_and_forward_returns_with_options,
};
pub use data::validate_factor_frame;
pub use error::{FerricAlphaError, Result};
pub use forward_returns::{
    ForwardReturnsOptions, compute_forward_returns, compute_forward_returns_with_options,
};
pub use performance::{
    AlphaBetaOptions, CommonStartReturnsOptions, EventReturnsOptions, IcOptions, MeanIcOptions,
    MeanReturnByQuantileOptions, PortfolioOptions, PyfolioInput, ReturnOptions, TimeBucket,
    WeightOptions, average_cumulative_return_by_quantile,
    average_cumulative_return_by_quantile_from_prices, common_start_returns,
    compute_mean_returns_spread, create_pyfolio_input, cumulative_returns, factor_alpha_beta,
    factor_cumulative_returns, factor_information_coefficient, factor_positions,
    factor_rank_autocorrelation, factor_returns, factor_weights, mean_information_coefficient,
    mean_return_by_quantile, positions, quantile_turnover,
};
pub use quantiles::{QuantizeOptions, quantize_factor};
pub use report::{
    ColumnRole, DisplayFormat, DisplayUnit, EventReturnsReportData, EventReturnsTearSheetData,
    EventStudyReportData, EventStudyTearSheetData, EventStudyTearSheetOptions, FullReportData,
    FullTearSheetData, FullTearSheetOptions, InformationReportData, InformationTearSheetData,
    InformationTearSheetOptions, ReportColumn, ReportColumnData, ReportKind, ReportMetadata,
    ReportTable, ReportTimeUnit, ReturnsReportData, ReturnsTearSheetData, ReturnsTearSheetOptions,
    SummaryReportData, SummaryTearSheetData, SummaryTearSheetOptions, TearSheetData,
    TurnoverReportData, TurnoverTearSheetData, TurnoverTearSheetOptions,
    create_event_returns_tear_sheet_data, create_event_study_tear_sheet_data,
    create_full_tear_sheet_data, create_information_tear_sheet_data,
    create_returns_tear_sheet_data, create_summary_tear_sheet_data,
    create_turnover_tear_sheet_data, factor_quantile_statistics,
};
pub use statistics::{
    LinearFit, QqPoint, TTest, average_rank, biased_excess_kurtosis, biased_skew, normal_qq,
    one_sample_t_test, pearson, sample_std, simple_ols, spearman, std_error,
};

#[cfg(test)]
mod tests {
    #[test]
    fn crate_version_matches_package_version() {
        assert_eq!(super::VERSION, env!("CARGO_PKG_VERSION"));
    }
}
