use ennbo_bpann::BpannError;
use ndarray::Array2;

#[test]
fn invalid_parameter_and_shape_display() {
    let err = BpannError::InvalidParameter("bad".into());
    assert!(err.to_string().contains("bad"));
    let err = BpannError::InvalidShape {
        expected: vec![1, 2],
        got: vec![3, 4],
    };
    assert!(err.to_string().contains("1"));
}

#[test]
fn from_shape_error_maps_to_invalid_parameter() {
    let shape_err = Array2::<f64>::zeros((2, 2))
        .into_shape_with_order((1, 5))
        .unwrap_err();
    let mapped = BpannError::from(shape_err);
    assert!(matches!(mapped, BpannError::InvalidParameter(_)));
}
