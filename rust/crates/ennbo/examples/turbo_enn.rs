use ndarray::array;
use ennbo::{create_optimizer_enn, turbo_enn_config};

fn main() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let mut opt = create_optimizer_enn(bounds, None, None, 7).unwrap();
    let x = opt.ask(2).unwrap();
    let y = array![[x[[0, 0]]], [x[[1, 0]]]];
    opt.tell(&x.view(), &y.view(), None).unwrap();
    println!("incumbent={:?}", opt.incumbent_x());
    let _ = turbo_enn_config();
}
