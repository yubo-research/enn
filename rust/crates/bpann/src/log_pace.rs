pub(crate) const KICK_BYTES: usize = 1 << 20;
pub(crate) const DROP_ALIGN_BYTES: usize = 2 << 20;

pub(crate) fn enabled() -> bool {
    static ENV: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENV.get_or_init(|| std::env::var("ENN_LOG_PACE").as_deref() != Ok("0"))
}

pub(crate) fn kick(fd: i32, kicked: &mut usize, head: usize) {
    let end = head / KICK_BYTES * KICK_BYTES;
    if end <= *kicked {
        return;
    }
    #[cfg(target_os = "linux")]
    unsafe {
        libc::sync_file_range(
            fd,
            *kicked as libc::off64_t,
            (end - *kicked) as libc::off64_t,
            libc::SYNC_FILE_RANGE_WRITE,
        );
    }
    *kicked = end;
}

pub(crate) fn drop_range(
    dropped: usize,
    head: usize,
    lag: usize,
    step: usize,
) -> Option<(usize, usize)> {
    let upto = head.saturating_sub(lag) / DROP_ALIGN_BYTES * DROP_ALIGN_BYTES;
    (upto >= dropped + step.max(DROP_ALIGN_BYTES)).then_some((dropped, upto))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kick_advances_to_whole_chunks_only() {
        let mut k = 0;
        kick(-1, &mut k, KICK_BYTES - 1);
        assert_eq!(k, 0);
        kick(-1, &mut k, 3 * KICK_BYTES + 5);
        assert_eq!(k, 3 * KICK_BYTES);
    }

    #[test]
    fn drop_range_is_aligned_and_waits_for_a_step() {
        let (lag, step) = (16 << 20, 4 << 20);
        assert_eq!(drop_range(0, lag + (3 << 20), lag, step), None);
        let (a, b) = drop_range(0, lag + (5 << 20), lag, step).unwrap();
        assert_eq!((a, b), (0, 4 << 20));
        assert_eq!(b % DROP_ALIGN_BYTES, 0);
        assert_eq!(drop_range(b, lag + (7 << 20), lag, step), None);
    }
}
