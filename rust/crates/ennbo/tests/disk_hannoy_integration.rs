//! Disk Hannoy integration: incremental add, sync, search.

#![cfg(feature = "hannoy")]

mod disk_streaming_helper;

use disk_streaming_helper::run_disk_streaming_add_sync_search;
use ennbo::IndexDriver;

#[test]
fn disk_hannoy_10k_add_sync_search() {
    run_disk_streaming_add_sync_search(IndexDriver::HNSWHannoy);
}
