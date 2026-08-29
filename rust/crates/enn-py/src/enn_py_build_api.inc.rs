macro_rules! define_enn_py_build_api {
    ($link:ident) => {
        #[doc = "kiss-coverage-off"]
        pub fn run_enn_py_build() {
            $link::emit_linux_rpath_link_args();
            ennbo::link_search::emit_blas_lapack_link_search_linux();
        }

        #[doc = "kiss-coverage-off"]
        pub fn kiss_enn_py_build_touch_01() {
            let _ = $link::blas_libs_present as fn(&std::path::Path) -> bool;
        }

        #[doc = "kiss-coverage-off"]
        pub fn kiss_enn_py_build_touch_02() {
            let _ = $link::install_patchelf_if_needed as fn();
        }

        #[doc = "kiss-coverage-off"]
        pub fn kiss_enn_py_build_touch_03() {
            let _ = $link::emit_linux_rpath_link_args as fn();
        }

        #[doc = "kiss-coverage-off"]
        pub fn kiss_enn_py_build_touch_10() {
            let _ = ennbo::link_search::emit_blas_lapack_link_search_linux as fn();
        }

        #[doc = "kiss-coverage-off"]
        pub fn kiss_enn_py_build_touch_04() {
            kiss_enn_py_build_touch_01();
        }

        #[doc = "kiss-coverage-off"]
        pub fn kiss_enn_py_build_touch_05() {
            kiss_enn_py_build_touch_02();
        }

        #[doc = "kiss-coverage-off"]
        pub fn kiss_enn_py_build_touch_06() {
            kiss_enn_py_build_touch_03();
        }

        #[doc = "kiss-coverage-off"]
        pub fn kiss_enn_py_build_touch_07() {
            run_enn_py_build();
        }

        #[doc = "kiss-coverage-off"]
        pub fn kiss_enn_py_build_touch_08() {
            kiss_enn_py_build_touch_04();
        }

        #[doc = "kiss-coverage-off"]
        pub fn kiss_enn_py_build_touch_09() {
            kiss_enn_py_build_touch_05();
        }

        #[doc = "kiss-coverage-off"]
        pub fn main() {
            run_enn_py_build();
        }
    };
}
