//! Categorical layouts and validation, independent of the language binding.
use crate::{categorical, error::ENNError};

/// Representation of one categorical variable.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CatEncoding {
    /// One coordinate per category.
    OneHot,
    /// Fixed-width deterministic sign vectors.
    Rademacher,
    /// Choose the shorter block.
    Auto,
}

/// One block in encoded rows.
#[derive(Debug, Clone)]
pub struct CatBlock {
    /// Input column index.
    pub source: usize,
    /// Output block width.
    pub width: usize,
    /// Continuous, onehot, or code.
    pub style: &'static str,
    /// Categorical variable index for deterministic hashing.
    pub variable: usize,
}

/// Options for constructing a categorical input representation.
#[derive(Debug, Clone)]
pub struct CatOptions {
    /// Categorical representation policy.
    pub encoding: CatEncoding,
    /// Fixed Rademacher width, excluding cards and hash_dim.
    pub code_dim: Option<usize>,
    /// Shared hash width, excluding cards and code_dim.
    pub hash_dim: Option<usize>,
    /// Seed shared across deterministic category hashes.
    pub seed: u64,
    /// Maximum width for automatically chosen Rademacher blocks.
    pub dim_cap: usize,
    /// Norm of one categorical code.
    pub separation: f64,
}
impl Default for CatOptions {
    fn default() -> Self {
        Self {
            encoding: CatEncoding::Auto,
            code_dim: None,
            hash_dim: None,
            seed: 0,
            dim_cap: 48,
            separation: 10.0,
        }
    }
}

/// Choose a categorical block's width and style.
pub fn column_plan(
    card: usize,
    encoding: CatEncoding,
    cap: usize,
) -> Result<(usize, &'static str), ENNError> {
    if card == 0 {
        return Err(invalid("card must be >= 1"));
    }
    if encoding == CatEncoding::OneHot {
        return Ok((card, "onehot"));
    }
    let width = categorical::log_code_width(card, cap)?;
    Ok(if encoding == CatEncoding::Rademacher || card > width {
        (width, "code")
    } else {
        (card, "onehot")
    })
}

pub(crate) fn invalid(message: &str) -> ENNError {
    ENNError::InvalidParameter(message.into())
}

/// Validate layout options before constructing any blocks.
pub fn validate(
    kinds: &[bool],
    cards: Option<&[Option<usize>]>,
    options: &CatOptions,
) -> Result<(), ENNError> {
    if kinds.is_empty() {
        return Err(invalid("kinds must list at least one column"));
    }
    if !options.separation.is_finite() || options.separation <= 0.0 {
        return Err(invalid("separation must be positive and finite"));
    }
    if options.code_dim.is_some() && options.hash_dim.is_some() {
        return Err(invalid("pass only one of code_dim, hash_dim"));
    }
    validate_dimensions(options)?;
    if options.code_dim.is_some() || options.hash_dim.is_some() {
        if cards.is_some() {
            return Err(invalid(if options.code_dim.is_some() {
                "code_dim does not use cards"
            } else {
                "hash_dim does not use cards"
            }));
        }
        if !kinds.contains(&true) {
            return Err(invalid("no categorical columns to encode"));
        }
    } else {
        categorical::width(options.dim_cap, 64)?;
        validate_cards(kinds, cards)?;
    }
    Ok(())
}

fn validate_dimensions(options: &CatOptions) -> Result<(), ENNError> {
    if let Some(w) = options.code_dim {
        categorical::width(w, 64).map_err(|_| invalid("code_dim must be in 1..64"))?;
        if options.encoding != CatEncoding::Rademacher {
            return Err(invalid("code_dim is a Rademacher width"));
        }
    }
    if let Some(w) = options.hash_dim {
        categorical::width(w, usize::MAX).map_err(|_| invalid("hash_dim must be >= 1"))?;
        if options.encoding != CatEncoding::Auto {
            return Err(invalid("hash_dim already chooses the categorical encoding"));
        }
    }
    Ok(())
}

fn validate_cards(kinds: &[bool], cards: Option<&[Option<usize>]>) -> Result<(), ENNError> {
    let Some(cards) = cards else {
        return if kinds.contains(&true) {
            Err(invalid("categorical columns need cards"))
        } else {
            Ok(())
        };
    };
    if cards.len() != kinds.len() {
        return Err(invalid("cards has wrong length"));
    }
    for (&categorical, &card) in kinds.iter().zip(cards) {
        match (categorical, card) {
            (true, Some(0)) => return Err(invalid("card must be >= 1")),
            (true, None) => return Err(invalid("a categorical column needs a card")),
            (false, Some(_)) => return Err(invalid("a continuous column has no card")),
            _ => {}
        }
    }
    Ok(())
}

/// Construct output blocks in input-column order, retaining continuous columns first for hashes.
pub fn blocks(
    kinds: &[bool],
    cards: Option<&[Option<usize>]>,
    options: &CatOptions,
) -> Result<Vec<CatBlock>, ENNError> {
    validate(kinds, cards, options)?;
    let mut blocks = Vec::new();
    let mut variable = 0;
    for (source, &categorical) in kinds.iter().enumerate() {
        if !categorical {
            blocks.push(CatBlock {
                source,
                width: 1,
                style: "continuous",
                variable: 0,
            });
        } else {
            if options.hash_dim.is_none() {
                let (width, style) = match options.code_dim {
                    Some(w) => (w, "code"),
                    None => column_plan(
                        cards.and_then(|c| c[source]).unwrap_or(0),
                        options.encoding,
                        options.dim_cap,
                    )?,
                };
                blocks.push(CatBlock {
                    source,
                    width,
                    style,
                    variable,
                });
            }
            variable += 1;
        }
    }
    Ok(blocks)
}
