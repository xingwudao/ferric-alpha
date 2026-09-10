use polars::prelude::DataFrame;

use crate::Result;
use crate::performance::frame::{FactorData, validate_factor_data};

pub(crate) struct ReportContext {
    factor_data: FactorData,
    normalized_factor: DataFrame,
    ic: Option<DataFrame>,
    means: Option<DataFrame>,
    returns: Option<DataFrame>,
    alpha_beta: Option<DataFrame>,
    turnover: Option<DataFrame>,
    rank_autocorrelation: Option<DataFrame>,
}

impl ReportContext {
    pub(crate) fn new(frame: &DataFrame) -> Result<Self> {
        Ok(Self {
            factor_data: validate_factor_data(frame, true, false, false)?,
            normalized_factor: frame.clone(),
            ic: None,
            means: None,
            returns: None,
            alpha_beta: None,
            turnover: None,
            rank_autocorrelation: None,
        })
    }

    pub(crate) fn factor_data(&self) -> &FactorData {
        &self.factor_data
    }

    #[allow(dead_code)]
    pub(crate) fn normalized_factor(&self) -> &DataFrame {
        &self.normalized_factor
    }

    #[allow(dead_code)]
    pub(crate) fn cached_tables(&self) -> [&Option<DataFrame>; 6] {
        [
            &self.ic,
            &self.means,
            &self.returns,
            &self.alpha_beta,
            &self.turnover,
            &self.rank_autocorrelation,
        ]
    }
}
