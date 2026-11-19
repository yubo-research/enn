#[path = "link_rpath.rs"]
mod link_rpath;

include!("enn_py_build_api.inc.rs");
define_enn_py_build_api!(link_rpath);
