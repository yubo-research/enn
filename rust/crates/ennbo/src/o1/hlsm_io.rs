pub(crate) fn par_map<T: Sync, R: Send>(items: &[T], width: usize, f: impl Fn(&T) -> R + Sync) -> Vec<R> {
    if width <= 1 || items.len() <= 1 {
        return items.iter().map(&f).collect();
    }
    let per = items.len().div_ceil(width);
    let f = &f;
    std::thread::scope(|s| {
        let parts: Vec<_> = items.chunks(per).map(|c| s.spawn(move || c.iter().map(f).collect::<Vec<R>>())).collect();
        parts.into_iter().flat_map(|h| h.join().expect("an O1E read thread panicked")).collect()
    })
}

#[cfg(test)]
mod tests {
    use super::par_map;

    #[test]
    fn par_map_keeps_order_at_every_width() {
        let v: Vec<u32> = (0..37).collect();
        for w in [1, 3, 8, 100] {
            assert_eq!(par_map(&v, w, |x| x * 2), v.iter().map(|x| x * 2).collect::<Vec<_>>());
        }
    }
}
