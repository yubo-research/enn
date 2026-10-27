//! CPU sentence embedding through sentence-transformers-rs; no Python runtime.
use crate::error::ENNError;
use ndarray::Array2;
use sentence_transformers_rs::sentence_transformer::{SentenceTransformerBuilder, Which};

fn preset(model: &str) -> Result<Which, ENNError> {
    use Which::*;
    [
        AllMiniLML6v2, AllMiniLML12v2, ParaphraseMiniLML6v2,
        ParaphraseMultilingualMiniLML12v2, LaBSE, ParaphraseMultilingualMpnetBaseV2,
        DistiluseBaseMultilingualCasedV2, BgeSmallEnV1_5, MultilingualE5Large,
        MultilingualE5Base, MultilingualE5Small, AllMpnetBaseV2,
        ParaphraseMpnetBaseV2, BgeBaseEnV1_5,
    ]
    .into_iter()
    .find(|which| which.to_string() == model)
    .ok_or_else(|| ENNError::InvalidParameter(format!(
        "unsupported sentence-transformers-rs model: {model}; use a supported model or supply encoder="
    )))
}

/// Embed a nonempty ordered batch using the model's pooling and dense presets.
/// The upstream loader downloads model files into the Hugging Face cache.
pub fn encode(texts: &[String], model: &str) -> Result<Array2<f64>, ENNError> {
    let which = preset(model)?;
    if texts.is_empty() {
        return Err(ENNError::InvalidParameter("texts must be nonempty".into()));
    }
    let encoder = SentenceTransformerBuilder::with_sentence_transformer(&which)
        .with_device(&candle_core::Device::Cpu)
        .build()
        .map_err(|e| ENNError::InvalidParameter(format!("loading {model}: {e}")))?;
    let lines: Vec<_> = texts.iter().map(String::as_str).collect();
    let embeddings = encoder
        .embed(&lines)
        .map_err(|e| ENNError::InvalidParameter(format!("embedding with {model}: {e}")))?;
    let width = embeddings.first().map_or(0, Vec::len);
    if width == 0 || embeddings.len() != texts.len() || embeddings.iter().any(|r| r.len() != width)
    {
        return Err(ENNError::InvalidParameter(
            "invalid sentence embeddings".into(),
        ));
    }
    Array2::from_shape_vec(
        (texts.len(), width),
        embeddings.into_iter().flatten().map(f64::from).collect(),
    )
    .map_err(|e| ENNError::InvalidParameter(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn invalid_requests_fail_before_model_download() {
        assert!(encode(&[], "sentence-transformers/all-MiniLM-L6-v2").is_err());
        assert!(encode(&["text".into()], "unknown/model")
            .unwrap_err()
            .to_string()
            .contains("unsupported"));
        assert!(preset("sentence-transformers/LaBSE").is_ok());
        assert!(preset("sentence-transformers/all-mpnet-base-v2").is_ok());
    }
}
