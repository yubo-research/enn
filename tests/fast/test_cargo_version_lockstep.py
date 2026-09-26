import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
WORKSPACE_MANIFESTS = [
    p for p in (REPO / "rust" / "Cargo.toml", REPO / "Cargo.toml") if p.exists()
]
DEP_TABLES = ("dependencies", "build-dependencies", "dev-dependencies")


def _load(path):
    return tomllib.loads(path.read_text())


def _workspace_version(manifest):
    return _load(manifest)["workspace"]["package"]["version"]


def _members(manifest):
    root = manifest.parent
    return {
        _load(root / m / "Cargo.toml")["package"]["name"]: root / m / "Cargo.toml"
        for m in _load(manifest)["workspace"]["members"]
    }


def _dep_tables(pkg):
    yield from (pkg.get(t, {}) for t in DEP_TABLES)
    for target in pkg.get("target", {}).values():
        yield from (target.get(t, {}) for t in DEP_TABLES)


def test_all_workspace_roots_share_one_version():
    versions = {str(m): _workspace_version(m) for m in WORKSPACE_MANIFESTS}
    assert len(set(versions.values())) == 1, versions


@pytest.mark.parametrize("manifest", WORKSPACE_MANIFESTS, ids=str)
def test_members_inherit_workspace_version(manifest):
    for name, path in _members(manifest).items():
        assert _load(path)["package"]["version"] == {"workspace": True}, name


@pytest.mark.parametrize("manifest", WORKSPACE_MANIFESTS, ids=str)
def test_internal_deps_pinned_exactly_to_workspace_version(manifest):
    version = _workspace_version(manifest)
    members = _members(manifest)
    ws_deps = _load(manifest)["workspace"].get("dependencies", {})
    for name, path in members.items():
        for table in _dep_tables(_load(path)):
            for dep in set(table) & set(members):
                assert table[dep] == {"workspace": True}, (name, dep)
                assert ws_deps[dep]["version"] == f"={version}", dep


@pytest.mark.parametrize("manifest", WORKSPACE_MANIFESTS, ids=str)
def test_lockfile_matches_workspace_version(manifest):
    lock = manifest.with_name("Cargo.lock")
    if not lock.exists():
        pytest.skip(f"{lock} not generated")
    version = _workspace_version(manifest)
    members = _members(manifest)
    locked = {
        p["name"]: p["version"]
        for p in _load(lock)["package"]
        if p["name"] in members
    }
    assert locked == dict.fromkeys(members, version)
