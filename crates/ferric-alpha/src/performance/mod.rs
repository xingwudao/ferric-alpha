mod alpha_beta;
mod common_start;
mod events;
pub(crate) mod frame;
mod ic;
mod portfolio;
mod pyfolio;
mod quantiles;
mod returns;
mod turnover;
mod weights;

pub use alpha_beta::factor_alpha_beta;
pub use common_start::{
    CommonStartReturnsOptions, average_cumulative_return_by_quantile_from_prices,
    common_start_returns,
};
pub(crate) use events::{EventCoverageRow, event_window_result};
pub use events::{EventReturnsOptions, average_cumulative_return_by_quantile};
pub use ic::{factor_information_coefficient, mean_information_coefficient};
pub use portfolio::{cumulative_returns, factor_cumulative_returns, factor_positions, positions};
pub use pyfolio::{PyfolioInput, create_pyfolio_input};
pub use quantiles::{compute_mean_returns_spread, mean_return_by_quantile};
pub use returns::factor_returns;
pub use turnover::{factor_rank_autocorrelation, quantile_turnover};
pub use weights::factor_weights;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeBucket {
    Daily,
    Weekly,
    Monthly,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct IcOptions {
    pub(crate) group_adjust: bool,
    pub(crate) by_group: bool,
}

impl IcOptions {
    pub fn with_group_adjust(mut self, group_adjust: bool) -> Self {
        self.group_adjust = group_adjust;
        self
    }

    pub fn with_by_group(mut self, by_group: bool) -> Self {
        self.by_group = by_group;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MeanIcOptions {
    pub(crate) group_adjust: bool,
    pub(crate) by_group: bool,
    pub(crate) by_time: Option<TimeBucket>,
}

impl MeanIcOptions {
    pub fn with_group_adjust(mut self, group_adjust: bool) -> Self {
        self.group_adjust = group_adjust;
        self
    }

    pub fn with_by_group(mut self, by_group: bool) -> Self {
        self.by_group = by_group;
        self
    }

    pub fn with_by_time(mut self, by_time: Option<TimeBucket>) -> Self {
        self.by_time = by_time;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WeightOptions {
    pub(crate) demeaned: bool,
    pub(crate) group_adjust: bool,
    pub(crate) equal_weight: bool,
}

impl Default for WeightOptions {
    fn default() -> Self {
        Self {
            demeaned: true,
            group_adjust: false,
            equal_weight: false,
        }
    }
}

impl WeightOptions {
    pub fn with_demeaned(mut self, demeaned: bool) -> Self {
        self.demeaned = demeaned;
        self
    }

    pub fn with_group_adjust(mut self, group_adjust: bool) -> Self {
        self.group_adjust = group_adjust;
        self
    }

    pub fn with_equal_weight(mut self, equal_weight: bool) -> Self {
        self.equal_weight = equal_weight;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ReturnOptions {
    pub(crate) weight_options: WeightOptions,
    pub(crate) by_asset: bool,
}

impl ReturnOptions {
    pub fn with_weight_options(mut self, weight_options: WeightOptions) -> Self {
        self.weight_options = weight_options;
        self
    }

    pub fn with_by_asset(mut self, by_asset: bool) -> Self {
        self.by_asset = by_asset;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AlphaBetaOptions {
    pub(crate) weight_options: WeightOptions,
}

impl AlphaBetaOptions {
    pub fn with_weight_options(mut self, weight_options: WeightOptions) -> Self {
        self.weight_options = weight_options;
        self
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeanReturnByQuantileOptions {
    pub(crate) by_date: bool,
    pub(crate) by_group: bool,
    pub(crate) demeaned: bool,
    pub(crate) group_adjust: bool,
}

impl Default for MeanReturnByQuantileOptions {
    fn default() -> Self {
        Self {
            by_date: false,
            by_group: false,
            demeaned: true,
            group_adjust: false,
        }
    }
}

impl MeanReturnByQuantileOptions {
    pub fn with_by_date(mut self, by_date: bool) -> Self {
        self.by_date = by_date;
        self
    }

    pub fn with_by_group(mut self, by_group: bool) -> Self {
        self.by_group = by_group;
        self
    }

    pub fn with_demeaned(mut self, demeaned: bool) -> Self {
        self.demeaned = demeaned;
        self
    }

    pub fn with_group_adjust(mut self, group_adjust: bool) -> Self {
        self.group_adjust = group_adjust;
        self
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct PortfolioOptions {
    pub(crate) weight_options: WeightOptions,
    pub(crate) quantiles: Option<Vec<u32>>,
    pub(crate) groups: Option<Vec<String>>,
}

impl PortfolioOptions {
    pub fn with_weight_options(mut self, weight_options: WeightOptions) -> Self {
        self.weight_options = weight_options;
        self
    }

    pub fn with_quantiles(mut self, quantiles: Option<Vec<u32>>) -> Self {
        self.quantiles = quantiles;
        self
    }

    pub fn with_groups(mut self, groups: Option<Vec<String>>) -> Self {
        self.groups = groups;
        self
    }
}
