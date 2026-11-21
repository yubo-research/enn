use std::cmp::Ordering;

#[derive(Copy, Clone, Debug)]
pub struct OrderedFloat<T = f32>(pub T);

pub use OrderedFloat as OrderedF32;

impl<T: PartialEq> PartialEq for OrderedFloat<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl<T: PartialEq> Eq for OrderedFloat<T> {}
impl<T: PartialOrd> PartialOrd for OrderedFloat<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<T: PartialOrd> Ord for OrderedFloat<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordered_float_orders_and_treats_nan_as_equal() {
        assert!(OrderedF32(1.0) > OrderedF32(0.0));
        assert_eq!(OrderedFloat(f64::NAN).cmp(&OrderedFloat(0.0)), Ordering::Equal);
    }
}
