#[test]
fn pymodule_hypervolume() {
    std::hint::black_box(enn_rust::pymodule_hypervolume);
}

#[test]
fn pymodule_hash() {
    std::hint::black_box(enn_rust::pymodule_hash);
}

#[test]
fn pymodule_util() {
    std::hint::black_box(enn_rust::pymodule_util);
}

#[test]
fn pymodule_model() {
    std::hint::black_box(enn_rust::pymodule_model);
}

#[test]
fn pymodule_fit() {
    std::hint::black_box(enn_rust::pymodule_fit);
}

#[test]
fn pymodule_optimizer() {
    std::hint::black_box(enn_rust::pymodule_optimizer);
}

#[test]
fn kiss_link_child_pymodule_exports() {
    enn_rust::kiss_link_child_pymodule_exports();
}
