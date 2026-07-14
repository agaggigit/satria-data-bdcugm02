import os

import pytest

from local_cache import localize_paths


def _make(tmp_path, rel, content=b"xxxxxxxxxx"):
    p = tmp_path / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(content)
    return str(p)


def test_returns_local_paths_mirroring_source_structure(tmp_path):
    files = [_make(tmp_path, f"drive/train/cls{i % 2}/img{i}.jpg") for i in range(4)]
    local = tmp_path / "local"

    out = localize_paths(files, str(local))

    assert len(out) == 4
    for p in out:
        assert os.path.exists(p)
        assert str(local) in p
    # struktur subfolder (cls0/cls1) ikut termirror, bukan diratakan
    assert any(os.path.join("cls0", "img0.jpg") in p for p in out)


def test_copied_content_is_identical(tmp_path):
    files = [_make(tmp_path, "drive/train/a/img0.jpg", content=b"HALO123")]
    out = localize_paths(files, str(tmp_path / "local"))
    assert open(out[0], "rb").read() == b"HALO123"


def test_order_is_preserved_exactly(tmp_path):
    # KRITIS: baris ke-i embedding harus tetap = baris ke-i folds.csv.
    # Kalau localize_paths menukar urutan, seluruh alignment rusak diam-diam.
    files = [_make(tmp_path, f"drive/train/a/img{i}.jpg") for i in range(10)]
    out = localize_paths(files, str(tmp_path / "local"))
    assert [os.path.basename(p) for p in out] == [os.path.basename(p) for p in files]


def test_second_call_does_not_recopy(tmp_path):
    files = [_make(tmp_path, "drive/train/a/img0.jpg")]
    local = str(tmp_path / "local")

    out1 = localize_paths(files, local)
    mtime_before = os.path.getmtime(out1[0])

    out2 = localize_paths(files, local)

    assert out1 == out2
    assert os.path.getmtime(out2[0]) == mtime_before, "file di-copy ulang padahal sudah ada"


def test_missing_source_raises_and_leaves_no_partial_file(tmp_path):
    files = [_make(tmp_path, "drive/train/a/img0.jpg"), str(tmp_path / "drive/train/a/hilang.jpg")]
    local = tmp_path / "local"

    with pytest.raises(OSError):
        localize_paths(files, str(local))

    # tidak boleh meninggalkan file setengah jadi yang nanti dikira valid
    assert not (local / "a" / "hilang.jpg").exists()
    assert not list(local.rglob("*.tmp"))
