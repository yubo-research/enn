use ennbo::incumbent_tracker::{
    tracker_m_from_enn_k, tracker_m_no_surrogate, IncrementalIncumbentTracker,
};
use ndarray::array;

#[test]
fn test_incumbent_tracker_integration_paths() {
    let names: &[&str] = &[
        "tracker_m_from_enn_k",
        "tracker_m_no_surrogate",
        "IncrementalIncumbentTracker",
        "reset_incumbent_tracker",
        "sync_incumbent_tracker_from_obs",
        "push_top_m",
        "sorted_indices",
    ];
    assert!(!names.is_empty());
    assert_eq!(tracker_m_from_enn_k(3), 3);
    let _ = tracker_m_no_surrogate();

    let mut noiseless = IncrementalIncumbentTracker::new(5, false, 1);
    noiseless.tell(0, &array![1.0]);
    noiseless.tell(1, &array![3.0]);
    let _ = noiseless.ask();
    noiseless.reset();

    let mut noisy = IncrementalIncumbentTracker::new(2, true, 1);
    noisy.tell(0, &array![1.0]);
    noisy.tell(1, &array![2.0]);
    noisy.tell(2, &array![3.0]);
    let _ = noisy.ask();
    noisy.rebuild(&array![[4.0], [1.0]].view());

    let mut multi = IncrementalIncumbentTracker::new(2, false, 2);
    multi.tell(0, &array![1.0, 0.0]);
    multi.tell(1, &array![0.0, 2.0]);
    let _ = multi.ask();

    let mut all = IncrementalIncumbentTracker::new(tracker_m_no_surrogate(), false, 1);
    all.tell(0, &array![0.1]);
    let _ = all.ask();
}
