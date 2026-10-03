//! Run the five `ennbo` example programs.
//!
//! `cargo nextest` does not execute example binaries. Including each example
//! and calling its `main` runs that program as a normal test.

mod affine_calibration {
    include!("../examples/affine_calibration.rs");

    #[test]
    fn execute() {
        main();
    }
}

mod bpann_disk_auto_metric {
    include!("../examples/bpann_disk_auto_metric.rs");

    #[test]
    fn execute() {
        main();
    }
}

mod enn_posterior {
    include!("../examples/enn_posterior.rs");

    #[test]
    fn execute() {
        main();
    }
}

mod morbo_enn {
    include!("../examples/morbo_enn.rs");

    #[test]
    fn execute() {
        main();
    }
}

mod turbo_enn {
    include!("../examples/turbo_enn.rs");

    #[test]
    fn execute() {
        main();
    }
}
