#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum FerricAlphaError {
    #[error("missing required column: {0}")]
    MissingColumn(&'static str),
    #[error("column {column} has invalid dtype: expected {expected}, got {actual}")]
    InvalidDtype {
        column: &'static str,
        expected: &'static str,
        actual: String,
    },
    #[error("date and asset must not contain null values")]
    NullKey,
    #[error("duplicate date/asset key")]
    DuplicateKey,
    #[error("period count must be positive, got {count}")]
    InvalidPeriod { count: u32 },
    #[error("periods must be unique")]
    DuplicatePeriod,
    #[error("trading calendar sessions must be sorted ascending")]
    UnsortedSessions,
    #[error("trading calendar sessions must be unique")]
    DuplicateSession,
    #[error("factor and price date timezones must match")]
    TimezoneMismatch,
    #[error("price must be positive and finite")]
    InvalidPrice,
    #[error("invalid quantile request: {reason}")]
    InvalidQuantiles { reason: &'static str },
    #[error("invalid input for {operation}: {reason}")]
    InvalidInput {
        operation: &'static str,
        reason: &'static str,
    },
    #[error("invalid report for {operation}: {reason}")]
    InvalidReport {
        operation: &'static str,
        reason: &'static str,
    },
    #[error("unsupported report contract version: expected {expected}, got {actual}")]
    ReportVersion {
        expected: &'static str,
        actual: String,
    },
    #[error("invalid option {option}: {reason}")]
    InvalidOption {
        option: &'static str,
        reason: &'static str,
    },
    #[error("quantiles and bins are mutually exclusive")]
    ConflictingQuantileMode,
    #[error("insufficient observations for {operation}")]
    InsufficientData { operation: &'static str },
    #[error("data loss {actual:.4} exceeded max_loss {allowed:.4}")]
    MaxLossExceeded { actual: f64, allowed: f64 },
    #[error("max_loss must be finite and between 0.0 and 1.0, got {value}")]
    InvalidMaxLoss { value: f64 },
    #[error(transparent)]
    Polars(#[from] polars::error::PolarsError),
}

pub type Result<T> = std::result::Result<T, FerricAlphaError>;
