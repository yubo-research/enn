#[path = "link_search.rs"]
mod link_search;

include!("ennbo_build_api.inc.rs");
define_ennbo_build_api!(link_search);
