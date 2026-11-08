#![allow(clippy::unreadable_literal, clippy::excessive_precision)]

#[path = "numpy_ziggurat_ki.rs"]
mod ki;
#[path = "numpy_ziggurat_wi.rs"]
mod wi;

pub use ki::KI_DOUBLE;
pub use wi::{FI_DOUBLE, WI_DOUBLE, ZIGGURAT_NOR_INV_R, ZIGGURAT_NOR_R};
