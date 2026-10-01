use ndarray::array;
use ennbo::{create_optimizer_enn_with_overrides, ConfigOverrides, TrustRegionKind};

fn main() {
    let bounds = array![[0.0, 1.0], [0.0, 1.0]];
    let overrides = ConfigOverrides {
        trust_region_kind: Some(TrustRegionKind::Morbo),
        num_metrics: Some(2),
        ..Default::default()
    };
    let mut opt =
        create_optimizer_enn_with_overrides(bounds, None, None, 3, Some(&overrides)).unwrap();
    let x = opt.ask(2).unwrap();
    let y = array![[0.1, 0.2], [0.3, 0.0]];
    opt.tell(&x.view(), &y.view(), None).unwrap();
    println!("rows={}", opt.obs_count());
}
