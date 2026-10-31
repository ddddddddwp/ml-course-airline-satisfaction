"""数据侧回归测试：去重口径 + 校验和归一化 + CHECKSUMS 闸门。

全部用函数级 + 临时小文件，不读 92 MB 的真实数据。
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import pytest

from src.common.config import (
    checksum_problems,
    read_checksums,
    sha256_file,
    verify_checksums,
    write_checksums,
)
from src.data.make_dataset import _checksum_gate, dedupe


# ------------------------------------------------------------------ 去重


def test_dedupe_ignores_unique_id() -> None:
    """id 唯一、特征行重复：旧实现 drop_duplicates() 一行都删不掉。"""
    df = pd.DataFrame({
        "id": [1, 2, 3],
        "Gender": ["Female", "Male", "Female"],
        "Age": [30, 40, 30],
        "satisfaction": [1, 0, 1],
    })
    out, removed, dup_ids = dedupe(df, "id")

    assert removed == 1
    assert list(out["id"]) == [1, 2]
    assert dup_ids == 0
    assert list(out.index) == [0, 1]


def test_dedupe_keeps_distinct_rows() -> None:
    df = pd.DataFrame({"id": [1, 2], "Age": [20, 21], "satisfaction": [0, 1]})
    out, removed, _ = dedupe(df, "id")

    assert removed == 0
    assert len(out) == 2


def test_dedupe_reports_duplicate_ids_without_dropping() -> None:
    """id 重复但特征不同：不按 id 删行，只统计并交给调用方告警。"""
    df = pd.DataFrame({"id": [1, 1], "Age": [20, 30], "satisfaction": [0, 1]})
    out, removed, dup_ids = dedupe(df, "id")

    assert dup_ids == 1
    assert removed == 0
    assert len(out) == 2


def test_dedupe_without_id_column_uses_all_columns() -> None:
    df = pd.DataFrame({"Age": [20, 20, 21], "satisfaction": [0, 0, 1]})
    out, removed, dup_ids = dedupe(df, None)

    assert removed == 1
    assert dup_ids == 0
    assert len(out) == 2


def test_dedupe_on_real_columns_does_not_use_id() -> None:
    """只有 id 不同、其余全同的两行应被删掉（这正是原始数据里最危险的重复）。"""
    rows = [
        {"id": 10, "Age": 35, "Class": "Eco", "satisfaction": 1},
        {"id": 11, "Age": 35, "Class": "Eco", "satisfaction": 1},
        {"id": 12, "Age": 36, "Class": "Eco", "satisfaction": 1},
    ]
    out, removed, _ = dedupe(pd.DataFrame(rows), "id")
    assert removed == 1
    assert list(out["id"]) == [10, 12]


# ------------------------------------------------------------------ 校验和归一化


def test_sha256_crlf_and_lf_csv_hash_same(tmp_path) -> None:
    text = "id,satisfaction\r\n1,1\r\n2,0\r\n"
    crlf = tmp_path / "train_crlf.csv"
    crlf.write_bytes(text.encode("utf-8"))
    lf = tmp_path / "train_lf.csv"
    lf.write_bytes(text.replace("\r\n", "\n").encode("utf-8"))

    expected = hashlib.sha256(text.replace("\r\n", "\n").encode("utf-8")).hexdigest()
    assert sha256_file(crlf) == sha256_file(lf) == expected

    # 关掉归一化后两者必须不同——证明摘要变化确实来自换行
    assert sha256_file(crlf, normalize_newlines=False) != sha256_file(lf, normalize_newlines=False)
    assert sha256_file(crlf, normalize_newlines=False) == hashlib.sha256(text.encode()).hexdigest()


def test_sha256_normalizes_across_chunk_boundaries(tmp_path) -> None:
    """CRLF 正好跨读块边界时也不能漏（65 MB 的 train.csv 靠分块读）。"""
    data = b"a\r\n" * 10 + b"b\rc"  # 末尾故意留一个孤立 \r
    p = tmp_path / "train.csv"
    p.write_bytes(data)
    expected = hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()

    for chunk in (1, 2, 3, 5, 8, 64, 1 << 20):
        assert sha256_file(p, chunk=chunk) == expected, chunk


def test_sha256_binary_files_are_not_normalized(tmp_path) -> None:
    p = tmp_path / "train.parquet"
    raw = b"\x00\r\n\x01binary"
    p.write_bytes(raw)
    assert sha256_file(p) == hashlib.sha256(raw).hexdigest()


# ------------------------------------------------------------------ CHECKSUMS


def _write_raw(tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "train.csv").write_bytes(b"id,satisfaction\r\n1,1\r\n2,0\r\n")
    (raw / "test.csv").write_bytes(b"id,Age\r\n3,30\r\n")
    return raw


def _files(raw: Path) -> list[Path]:
    return [raw / "train.csv", raw / "test.csv"]


def test_write_checksums_is_lf_and_verifies(tmp_path) -> None:
    raw = _write_raw(tmp_path)
    ck = raw / "CHECKSUMS.txt"
    write_checksums(ck, _files(raw))

    # Windows 上也必须写 LF，否则 sha256sum -c 会把 \r 当成文件名的一部分
    assert b"\r\n" not in ck.read_bytes()
    assert set(read_checksums(ck)) == {"train.csv", "test.csv"}

    report = verify_checksums(raw, _files(raw))
    assert report["ok"] is True
    assert not checksum_problems(report)
    assert set(report["verified"]) == {"train.csv", "test.csv"}


def test_verify_checksums_same_after_lf_checkout(tmp_path) -> None:
    """记录时工作区是 CRLF、之后（Linux/CI）变成 LF：仍应视为同一份数据。"""
    raw = _write_raw(tmp_path)
    write_checksums(raw / "CHECKSUMS.txt", _files(raw))

    (raw / "train.csv").write_bytes(b"id,satisfaction\n1,1\n2,0\n")
    report = verify_checksums(raw, _files(raw))
    assert report["ok"] is True


def test_verify_checksums_detects_replaced_data(tmp_path) -> None:
    raw = _write_raw(tmp_path)
    write_checksums(raw / "CHECKSUMS.txt", _files(raw))

    (raw / "train.csv").write_bytes(b"id,satisfaction\r\n1,0\r\n2,1\r\n")
    report = verify_checksums(raw, _files(raw))

    assert report["ok"] is False
    assert report["mismatched"] == ["train.csv"]
    assert any("train.csv" in p for p in checksum_problems(report))


def test_verify_checksums_reports_missing_and_absent(tmp_path) -> None:
    raw = _write_raw(tmp_path)
    write_checksums(raw / "CHECKSUMS.txt", [raw / "train.csv"])

    report = verify_checksums(raw, _files(raw))
    assert report["missing"] == ["test.csv"]
    assert report["ok"] is False

    (raw / "train.csv").unlink()
    report = verify_checksums(raw, [raw / "train.csv"])
    assert report["absent"] == ["train.csv"]
    assert report["ok"] is False


def test_read_checksums_tolerates_crlf_file(tmp_path) -> None:
    """旧版脚本在 Windows 上用默认换行写出过 CRLF 的 CHECKSUMS.txt。"""
    raw = _write_raw(tmp_path)
    ck = raw / "CHECKSUMS.txt"
    write_checksums(ck, _files(raw))
    ck.write_bytes(ck.read_bytes().replace(b"\n", b"\r\n"))

    report = verify_checksums(raw, _files(raw))
    assert report["ok"] is True


# ------------------------------------------------------------------ make_dataset 闸门


def test_checksum_gate_first_run_writes_file(tmp_path) -> None:
    raw = _write_raw(tmp_path)
    ck = raw / "CHECKSUMS.txt"
    assert not ck.exists()

    _checksum_gate(raw, raw / "train.csv", raw / "test.csv", accept=False)

    assert ck.exists()
    assert set(read_checksums(ck)) == {"train.csv", "test.csv"}


def test_checksum_gate_passes_and_does_not_rewrite(tmp_path) -> None:
    raw = _write_raw(tmp_path)
    tr, te = raw / "train.csv", raw / "test.csv"
    _checksum_gate(raw, tr, te, accept=False)
    ck = raw / "CHECKSUMS.txt"
    before, mtime = ck.read_bytes(), ck.stat().st_mtime_ns

    _checksum_gate(raw, tr, te, accept=False)  # 一致：继续，不重写

    assert ck.read_bytes() == before
    assert ck.stat().st_mtime_ns == mtime


def test_checksum_gate_rejects_replaced_data(tmp_path) -> None:
    raw = _write_raw(tmp_path)
    tr, te = raw / "train.csv", raw / "test.csv"
    _checksum_gate(raw, tr, te, accept=False)

    tr.write_bytes(b"id,satisfaction\r\n1,0\r\n2,1\r\n")
    with pytest.raises(SystemExit) as excinfo:
        _checksum_gate(raw, tr, te, accept=False)
    assert "--accept-data-change" in str(excinfo.value)


def test_checksum_gate_accept_data_change_updates(tmp_path) -> None:
    raw = _write_raw(tmp_path)
    tr, te = raw / "train.csv", raw / "test.csv"
    _checksum_gate(raw, tr, te, accept=False)

    tr.write_bytes(b"id,satisfaction\r\n1,0\r\n2,1\r\n")
    _checksum_gate(raw, tr, te, accept=True)

    report = verify_checksums(raw, [tr, te])
    assert report["ok"] is True
    assert not checksum_problems(report)
