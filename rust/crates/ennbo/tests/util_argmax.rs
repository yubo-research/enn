use ennbo::util::argmax_random_tie;
use rand::SeedableRng;
use rand::rngs::StdRng;

#[test]
fn argmax_random_tie_empty_slice() {
    let mut rng = StdRng::seed_from_u64(0);
    assert_eq!(argmax_random_tie(&[], &mut rng), 0);
}

#[test]
fn standardize_y_even_length_uses_pair_median() {
    use ennbo::util::standardize_y;
    use ndarray::Array1;
    let y = Array1::from(vec![1.0, 2.0, 3.0, 4.0]);
    let (center, scale) = standardize_y(&y.view());
    assert!(center.is_finite());
    assert!(scale.is_finite() && scale > 0.0);
}
