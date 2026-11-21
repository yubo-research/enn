#[path = "link_rpath.rs"]
mod link_rpath;

mod enn_py_build_api;
enn_py_build_api::define_enn_py_build_api!(link_rpath);
