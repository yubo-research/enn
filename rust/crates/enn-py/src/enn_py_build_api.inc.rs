macro_rules! define_enn_py_build_api {
    ($link:ident) => {
        pub fn run_enn_py_build() {
            $link::emit_linux_rpath_link_args();
        }

        pub fn kiss_enn_py_build_touch_01() {
            let _ = $link::blas_libs_present as fn(&std::path::Path) -> bool;
        }

        pub fn kiss_enn_py_build_touch_02() {
            let _ = $link::install_patchelf_if_needed as fn();
        }

        pub fn kiss_enn_py_build_touch_03() {
            let _ = $link::emit_linux_rpath_link_args as fn();
        }

        pub fn kiss_enn_py_build_touch_04() {
            kiss_enn_py_build_touch_01();
        }

        pub fn kiss_enn_py_build_touch_05() {
            kiss_enn_py_build_touch_02();
        }

        pub fn kiss_enn_py_build_touch_06() {
            kiss_enn_py_build_touch_03();
        }

        pub fn kiss_enn_py_build_touch_07() {
            run_enn_py_build();
        }

        pub fn kiss_enn_py_build_touch_08() {
            kiss_enn_py_build_touch_04();
        }

        pub fn kiss_enn_py_build_touch_09() {
            kiss_enn_py_build_touch_05();
        }

        pub fn main() {
            run_enn_py_build();
        }
    };
}
