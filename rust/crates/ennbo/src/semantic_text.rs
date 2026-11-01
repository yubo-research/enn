use crate::error::ENNError;
use ndarray::{Array2, ArrayView2, Axis};
use std::collections::HashMap;

pub fn prepare(
    names: &[String],
    descriptions: Option<&[Option<String>]>,
) -> Result<(Vec<String>, Vec<usize>), ENNError> {
    if names.is_empty() || descriptions.is_some_and(|d| d.len() != names.len()) {
        return Err(ENNError::InvalidParameter(
            "one description is required per variable".into(),
        ));
    }
    let texts = names
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let description = descriptions.and_then(|d| d[i].as_deref()).unwrap_or("");
            Ok(format!("{name}::{}", description_text(description)?))
        })
        .collect::<Result<Vec<_>, ENNError>>()?;
    Ok(deduplicate(texts))
}

pub fn deduplicate(texts: Vec<String>) -> (Vec<String>, Vec<usize>) {
    let mut unique = Vec::new();
    let mut indices = HashMap::new();
    let mut rows = Vec::with_capacity(texts.len());
    for text in texts {
        let next = unique.len();
        let index = *indices.entry(text.clone()).or_insert_with(|| {
            unique.push(text);
            next
        });
        rows.push(index);
    }
    (unique, rows)
}

fn description_text(description: &str) -> Result<String, ENNError> {
    if !description.starts_with("file://") {
        return Ok(description.to_owned());
    }
    let uri =
        url::Url::parse(description).map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
    let path = uri
        .to_file_path()
        .map_err(|_| ENNError::InvalidParameter("description files must be local".into()))?;
    std::fs::read_to_string(path).map_err(|e| ENNError::InvalidParameter(e.to_string()))
}

pub fn project(
    embeddings: &ArrayView2<f64>,
    rows: &[usize],
    cutoff: f64,
) -> Result<Array2<f64>, ENNError> {
    if rows.is_empty() || rows.iter().copied().max() != embeddings.nrows().checked_sub(1) {
        return Err(ENNError::InvalidParameter(
            "encoder returned the wrong number of variables".into(),
        ));
    }
    crate::semantic_projection::projection(&embeddings.select(Axis(0), rows).view(), cutoff)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn deduplication_preserves_variable_rows_and_geometry() {
        let names = vec!["a".into(), "b".into(), "c".into()];
        let descriptions = vec![Some(" shared ".into()), None, Some("shared".into())];
        let (texts, rows) = prepare(&names, Some(&descriptions)).unwrap();
        assert_eq!(texts, ["a:: shared ", "b::", "c::shared"]);
        assert_eq!(rows, [0, 1, 2]);
        let (texts, rows) = deduplicate(vec!["x".into(), "y".into(), "x".into()]);
        assert_eq!(texts, ["x", "y"]);
        assert_eq!(rows, [0, 1, 0]);
        let rows = vec![0, 1, 0];
        let p = project(&array![[1., 0.], [0., 1.]].view(), &rows, 1.0).unwrap();
        assert!((&p.row(0) - &p.row(2)).iter().all(|v| v.abs() < 1e-12));
        assert_eq!(p.dim(), (3, 1));
        assert!(prepare(&names, Some(&[])).is_err());
        assert!(project(&array![[1., 0.]].view(), &rows, 1.0).is_err());
    }

    #[test]
    fn local_files_support_escaped_paths_and_reject_remote_hosts() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("description with spaces.txt");
        std::fs::write(&path, "  semantic description \n").unwrap();
        let uri = url::Url::from_file_path(&path).unwrap();
        assert_eq!(
            description_text(uri.as_str()).unwrap(),
            "  semantic description \n"
        );
        assert!(description_text("file://remote/tmp/text").is_err());
        assert!(description_text("file:///nonexistent-enn-description").is_err());
    }
}
