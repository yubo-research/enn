//! Preserve categorical integer identity while allowing continuous float coordinates.
use crate::categorical_plan::invalid;
use crate::error::ENNError;

pub(crate) trait CatValue: Copy {
    fn number(self) -> f64;
    fn category(self) -> Result<i64, ENNError>;
}
impl CatValue for i64 {
    fn number(self) -> f64 {
        self as f64
    }
    fn category(self) -> Result<i64, ENNError> {
        Ok(self)
    }
}
impl CatValue for f64 {
    fn number(self) -> f64 {
        self
    }
    fn category(self) -> Result<i64, ENNError> {
        if !self.is_finite()
            || self.fract() != 0.0
            || self < i64::MIN as f64
            || self >= -(i64::MIN as f64)
        {
            return Err(invalid(
                "categorical values must be integers in the i64 range",
            ));
        }
        Ok(self as i64)
    }
}

/// One mixed input cell, preserving exact category ids beside continuous values.
#[derive(Clone, Copy)]
pub enum InputValue {
    /// Continuous coordinate or an integer-valued floating category.
    Number(f64),
    /// Exact categorical label.
    Category(i64),
}
impl CatValue for InputValue {
    fn number(self) -> f64 {
        match self {
            Self::Number(value) => value,
            Self::Category(value) => value as f64,
        }
    }
    fn category(self) -> Result<i64, ENNError> {
        match self {
            Self::Number(value) => value.category(),
            Self::Category(value) => Ok(value),
        }
    }
}
