macro_rules! define_ennbo_build_api {
    ($link:ident) => {
        pub fn run_ennbo_build() {
            $link::emit_faiss_link_search();
        }

        pub fn kiss_ennbo_build_touch_01() {
            let _ = $link::emit_link_search as fn(&str);
        }

        pub fn kiss_ennbo_build_touch_02() {
            let _ = $link::has_faiss_c as fn(&std::path::Path) -> bool;
        }

        pub fn kiss_ennbo_build_touch_03() {
            let _ = $link::has_blas_for_link as fn(&std::path::Path) -> bool;
        }

        pub fn kiss_ennbo_build_touch_04() {
            let _ = $link::openblas_for_link as fn(&std::path::Path) -> Option<std::path::PathBuf>;
        }

        pub fn kiss_ennbo_build_touch_05() {
            let _ = $link::emit_openblas_link as fn(&std::path::Path);
        }

        pub fn kiss_ennbo_build_touch_06() {
            let _ = $link::emit_blas_lapack_link_search_linux as fn();
        }

        pub fn kiss_ennbo_build_touch_07() {
            let _ = $link::emit_faiss_link_search as fn();
        }

        pub fn kiss_ennbo_build_touch_08() {
            kiss_ennbo_build_touch_01();
        }

        pub fn kiss_ennbo_build_touch_09() {
            run_ennbo_build();
        }

        pub fn main() {
            run_ennbo_build();
        }
    };
}
