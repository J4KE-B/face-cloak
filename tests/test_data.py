import pytest

from facecloak.data import DEFAULT_MANIFEST, load_manifest, shard


def test_shard_partition_and_determinism():
    items = [{"id": f"s{i:04d}"} for i in range(23)][::-1]  # unsorted input
    parts = [shard(items, f"{i}/4") for i in range(4)]
    ids = [x["id"] for p in parts for x in p]
    assert sorted(ids) == sorted(x["id"] for x in items)
    assert len(ids) == len(set(ids))
    assert parts == [shard(list(reversed(items)), f"{i}/4") for i in range(4)]
    assert shard(items, "0/1") == sorted(items, key=lambda x: x["id"])


def test_shard_plain_values_and_bad_spec():
    assert shard(["b", "a", "c"], "1/2") == ["b"]
    for bad in ("2/2", "0/0", "-1/3"):
        with pytest.raises(ValueError):
            shard(["a"], bad)


@pytest.fixture(scope="module")
def manifest():
    if not DEFAULT_MANIFEST.is_file():
        pytest.skip("manifest.json not generated")
    return load_manifest()


def test_manifest_schema(manifest):
    for k in ("version", "seed", "lfw_variant", "sources", "targets", "pairs"):
        assert k in manifest
    for s in manifest["sources"]:
        assert {"id", "identity", "src_path", "ref_path", "src_kps", "src_bbox", "ref_kps"} <= s.keys()
        assert len(s["src_kps"]) == 5 and len(s["ref_kps"]) == 5 and len(s["src_bbox"]) == 4
        assert not s["src_path"].startswith("/")
    for t in manifest["targets"]:
        assert {"id", "identity", "path", "kps", "bbox"} <= t.keys()
    src_ids = {s["identity"] for s in manifest["sources"]}
    assert not src_ids & {t["identity"] for t in manifest["targets"]}


def test_pairs_reference_existing_ids(manifest):
    sids = {s["id"] for s in manifest["sources"]}
    tids = {t["id"] for t in manifest["targets"]}
    assert manifest["pairs"]
    assert all(a in sids and b in tids for a, b in manifest["pairs"])
    assert sids == {a for a, _ in manifest["pairs"]}
