use crate::link_rpath as link;

pub fn run_enn_py_build() {
    link::emit_linux_rpath_link_args();
    ennbo::link_search::emit_blas_lapack_link_search_linux();
}

pub fn main() {
    run_enn_py_build();
}
