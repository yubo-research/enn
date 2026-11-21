#[path = "link_search.rs"]
mod link_search;

mod ennbo_build_api;
ennbo_build_api::define_ennbo_build_api!(link_search);
