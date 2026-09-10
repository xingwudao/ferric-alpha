use std::path::PathBuf;

use thiserror::Error;

pub type RenderResult<T> = Result<T, RenderError>;

#[derive(Debug, Error)]
pub enum RenderError {
    #[error("invalid render option {option}: {reason}")]
    InvalidOption {
        option: &'static str,
        reason: &'static str,
    },

    #[error("missing report table {id}")]
    MissingTable { id: String },

    #[error("missing column {column} in report table {table}")]
    MissingColumn { table: String, column: String },

    #[error("invalid column type for {table}.{column}: expected {expected}")]
    InvalidColumnType {
        table: String,
        column: String,
        expected: &'static str,
    },

    #[error("invalid render contract: {reason}")]
    InvalidContract { reason: String },

    #[error("render layout exceeds resource limits")]
    LayoutOverflow,

    #[error("render buffer allocation failed for {bytes} bytes")]
    Allocation { bytes: usize },

    #[error("rendered {format} output exceeds {limit} bytes")]
    OutputTooLarge { format: &'static str, limit: usize },

    #[error("{format} backend failed: {message}")]
    Backend {
        format: &'static str,
        message: String,
    },

    #[error("PNG encoding failed: {message}")]
    PngEncoding { message: String },

    #[error("failed to write {path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}
