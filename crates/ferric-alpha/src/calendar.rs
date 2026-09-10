use std::collections::HashSet;

use crate::error::{FerricAlphaError, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Period {
    count: u32,
}

impl Period {
    pub fn new(count: u32) -> Result<Self> {
        if count == 0 {
            return Err(FerricAlphaError::InvalidPeriod { count });
        }

        Ok(Self { count })
    }

    pub fn count(self) -> u32 {
        self.count
    }

    pub fn label(self) -> String {
        format!("{}D", self.count)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TradingCalendar {
    sessions: Vec<i64>,
    timezone: Option<String>,
}

impl TradingCalendar {
    pub fn new(sessions: Vec<i64>, timezone: Option<String>) -> Result<Self> {
        let mut seen = HashSet::with_capacity(sessions.len());
        let mut previous = None;
        for session in &sessions {
            if let Some(previous) = previous
                && previous > *session
            {
                return Err(FerricAlphaError::UnsortedSessions);
            }
            if !seen.insert(*session) {
                return Err(FerricAlphaError::DuplicateSession);
            }
            previous = Some(*session);
        }

        Ok(Self { sessions, timezone })
    }

    pub fn offset(&self, date: i64, period: Period) -> Option<i64> {
        let index = self.sessions.binary_search(&date).ok()?;
        self.sessions.get(index + period.count() as usize).copied()
    }

    pub fn timezone(&self) -> Option<&str> {
        self.timezone.as_deref()
    }

    pub(crate) fn sessions(&self) -> &[i64] {
        &self.sessions
    }
}

pub(crate) fn validate_periods(periods: &[Period]) -> Result<Vec<Period>> {
    if periods.is_empty() {
        return Err(FerricAlphaError::InsufficientData {
            operation: "periods",
        });
    }

    let mut sorted = periods.to_vec();
    sorted.sort();
    sorted.dedup();
    if sorted.len() != periods.len() {
        return Err(FerricAlphaError::DuplicatePeriod);
    }

    Ok(sorted)
}
