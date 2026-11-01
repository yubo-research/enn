use crate::adaptive_basis::Metadata;
use crate::adaptive_weight::PRIOR_WEIGHT;

pub(crate) fn geometries(
    metadata: &Metadata,
    adaptation: &str,
    weight: Option<f64>,
    scaled: bool,
) -> Vec<(&'static str, f64)> {
    let mut choices = vec![("fixed", 0.0)];
    if adaptation == "outputs" || adaptation == "none" {
        return choices;
    }
    if metadata.input.is_some() {
        choices.insert(0, ("raw", 0.0));
    }
    if scaled && metadata.input.as_ref().is_some_and(|p| p.ncols() > 0) {
        choices.insert(choices.len().min(2), ("scales", 1.0));
    }
    for geometry in ["linear", "quadratic"] {
        choices.push((geometry, weight.unwrap_or(PRIOR_WEIGHT)));
    }
    choices
}

pub(crate) fn allowed_output(name: &str, adaptation: &str, has_metadata: bool) -> bool {
    match adaptation {
        "inputs" => name == "independent",
        "none" => {
            name == if has_metadata {
                "semantic"
            } else {
                "independent"
            }
        }
        _ => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn directional(choices: &[(&str, f64)], name: &str) -> Vec<f64> {
        choices
            .iter()
            .filter(|choice| choice.0 == name)
            .map(|choice| choice.1)
            .collect()
    }

    #[test]
    fn linear_and_quadratic_are_scored_at_one_weight() {
        let open = geometries(&Metadata::default(), "all", None, false);
        assert_eq!(directional(&open, "linear"), vec![PRIOR_WEIGHT]);
        assert_eq!(directional(&open, "quadratic"), vec![PRIOR_WEIGHT]);
        let pinned = geometries(&Metadata::default(), "all", Some(30.0), false);
        assert_eq!(directional(&pinned, "linear"), vec![30.0]);
        assert_eq!(directional(&pinned, "quadratic"), vec![30.0]);
    }
}
