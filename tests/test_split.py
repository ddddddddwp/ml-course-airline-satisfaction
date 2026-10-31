"""划分与冻结的回归测试。

覆盖：
- build_folds：全覆盖 / 无交集 / 分层（各折正例率接近）
- freeze：默认不覆盖（exit 0 跳过）、--force 先备份再写
- load_folds：坏文件必须报错（长度、取值、空折、缺字段、非 JSON）
- verify_frozen：数据指纹一致 / 不一致 / 旧文件无指纹时跳过
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.common.config import sha256_file
from src.common.split import (
    build_folds,
    freeze,
    load_folds,
    verify_frozen,
    write_fingerprint,
)


def _toy_df(n: int = 2000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "id": np.arange(n),
        "Age": rng.integers(18, 70, n),
        "Flight Distance": rng.integers(50, 5000, n),
        "satisfaction": rng.binomial(1, 0.45, n).astype("int8"),
    })


def _cfg(tmp_path: Path, **split_over) -> dict:
    split = {
        "strategy": "stratified",
        "n_splits": 5,
        "seed": 42,
        "time_col": None,
        "group_col": "distance_bins",
        "group_bins": 20,
    }
    split.update(split_over)
    return {
        "project": "toy",
        "paths": {
            "processed_dir": str(tmp_path / "processed"),
            "splits": str(tmp_path / "splits" / "fold_v1.json"),
        },
        "data": {"target": "satisfaction", "id_col": "id"},
        "split": split,
    }


def _write_processed(cfg: dict, df: pd.DataFrame) -> Path:
    proc = Path(cfg["paths"]["processed_dir"])
    proc.mkdir(parents=True, exist_ok=True)
    p = proc / "train.parquet"
    df.to_parquet(p, index=False)
    return p


# ------------------------------------------------------------------ build_folds


def test_build_folds_covers_all_rows_without_overlap() -> None:
    df = _toy_df()
    cfg = _cfg(Path("."))
    fold = build_folds(df, cfg)

    assert fold.shape == (len(df),)
    assert set(np.unique(fold)) == set(range(cfg["split"]["n_splits"]))
    counts = [int((fold == f).sum()) for f in range(cfg["split"]["n_splits"])]
    assert sum(counts) == len(df)
    assert min(counts) > 0

    for f in range(cfg["split"]["n_splits"]):
        train_idx, valid_idx = np.where(fold != f)[0], np.where(fold == f)[0]
        assert len(np.intersect1d(train_idx, valid_idx)) == 0
        assert len(train_idx) + len(valid_idx) == len(df)


def test_build_folds_stratified_keeps_positive_rate_close() -> None:
    df = _toy_df(n=2000)
    cfg = _cfg(Path("."))
    fold = build_folds(df, cfg)
    y = df["satisfaction"].to_numpy()

    rates = [float(y[fold == f].mean()) for f in range(cfg["split"]["n_splits"])]
    overall = float(y.mean())
    assert max(abs(r - overall) for r in rates) < 0.02, rates
    assert max(rates) - min(rates) < 0.02


# ------------------------------------------------------------------ freeze


def test_freeze_writes_meta_without_overwriting_twice(tmp_path, capsys) -> None:
    cfg = _cfg(tmp_path)
    df = _toy_df()
    tr_path = _write_processed(cfg, df)
    out = freeze(cfg)

    assert out is not None and out.exists()
    meta = json.loads(out.read_text(encoding="utf-8"))
    assert meta["n_rows"] == len(df)
    assert meta["n_splits"] == 5
    assert len(meta["fold_of_row"]) == len(df)
    assert set(meta["fold_of_row"]) == set(range(5))
    assert meta["created_at"]
    assert meta["code_version"]
    assert meta["data_fingerprint"]["train.parquet"] == sha256_file(tr_path)

    # freeze 同时写旁路指纹：fold 文件之外还有一份可校验的快照
    sidecar = out.with_name(out.stem + ".fingerprint.json")
    assert sidecar.exists()
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["fold_file"] == out.name
    assert payload["fold_file_sha256"] == sha256_file(out, normalize_newlines=True)
    assert payload["n_rows"] == len(df)
    assert payload["n_splits"] == 5
    assert payload["data_fingerprint"] == meta["data_fingerprint"]
    assert payload["stamped_at"] and payload["code_version"]

    # 不带 --force 再来一次：不覆盖、退出码 0（make all 可重复执行）
    before = out.read_text(encoding="utf-8")
    again = freeze(cfg)
    assert again is None
    assert out.read_text(encoding="utf-8") == before
    assert "跳过" in capsys.readouterr().out
    assert not list(out.parent.glob("*.bak.*"))


def test_freeze_force_backs_up_old_file(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    df = _toy_df()
    _write_processed(cfg, df)
    out = freeze(cfg)
    before = out.read_text(encoding="utf-8")

    freeze(cfg, force=True)

    backups = sorted(out.parent.glob(out.name + ".bak.*"))
    assert len(backups) == 1, backups
    assert backups[0].read_text(encoding="utf-8") == before
    meta_new = json.loads(out.read_text(encoding="utf-8"))
    assert meta_new["n_rows"] == len(df)


def test_freeze_requires_processed_data(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    with pytest.raises(SystemExit):
        freeze(cfg)


# ------------------------------------------------------------------ load_folds


def test_load_folds_roundtrip(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    df = _toy_df()
    _write_processed(cfg, df)
    out = freeze(cfg)

    folds, meta = load_folds(cfg)
    assert out.exists()
    assert len(folds) == 5
    assert len(meta["fold_of_row"]) == len(df)
    for train_idx, valid_idx in folds:
        assert len(valid_idx) > 0
        assert len(np.intersect1d(train_idx, valid_idx)) == 0


BAD_METAS = [
    # fold_of_row 长度与 n_rows 不一致
    {"n_splits": 5, "n_rows": 6, "fold_of_row": [0, 1, 2, 3, 4]},
    # 取值越界
    {"n_splits": 4, "n_rows": 5, "fold_of_row": [0, 1, 2, 3, 4]},
    # 有折为空（3、4 折没有样本）
    {"n_splits": 5, "n_rows": 5, "fold_of_row": [0, 0, 1, 1, 2]},
    # 折数不合法
    {"n_splits": 1, "n_rows": 5, "fold_of_row": [0, 0, 0, 0, 0]},
    # 非整数取值
    {"n_splits": 5, "n_rows": 5, "fold_of_row": [0, 1, 2, 3, "x"]},
    # 缺 n_splits
    {"n_rows": 5, "fold_of_row": [0, 1, 2, 3, 4]},
    # 缺 fold_of_row
    {"n_splits": 5, "n_rows": 5},
]


@pytest.mark.parametrize("meta", BAD_METAS)
def test_load_folds_rejects_bad_meta(tmp_path, meta) -> None:
    p = tmp_path / "bad_fold.json"
    p.write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(SystemExit):
        load_folds(path=p)


def test_load_folds_rejects_invalid_json(tmp_path) -> None:
    p = tmp_path / "bad_fold.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(SystemExit):
        load_folds(path=p)


def test_load_folds_missing_file(tmp_path) -> None:
    with pytest.raises(SystemExit):
        load_folds(path=tmp_path / "nope.json")


# ------------------------------------------------------------------ verify_frozen


def test_verify_frozen_matches_fingerprint(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    _write_processed(cfg, _toy_df())
    freeze(cfg)

    report = verify_frozen(cfg)
    assert report["checked"] is True
    assert report["ok"] is True
    assert "数据指纹一致" in report["detail"]


def test_verify_frozen_detects_data_change(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    df = _toy_df()
    _write_processed(cfg, df)
    freeze(cfg)

    changed = df.copy()
    changed["Age"] = changed["Age"] + 1
    _write_processed(cfg, changed)

    report = verify_frozen(cfg)
    assert report["checked"] is True
    assert report["ok"] is False
    assert "指纹变化" in report["detail"]


def test_verify_frozen_skips_legacy_file_without_fingerprint(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    p = Path(cfg["paths"]["splits"])
    p.parent.mkdir(parents=True, exist_ok=True)
    legacy = {
        "version": "v1",
        "n_splits": 5,
        "n_rows": 10,
        "fold_of_row": [0, 1, 2, 3, 4] * 2,
    }
    p.write_text(json.dumps(legacy), encoding="utf-8")

    report = verify_frozen(cfg)
    assert report["checked"] is False
    assert report["ok"] is True
    assert "data_fingerprint" in report["detail"]


def test_stamp_enables_fingerprint_check_for_legacy_fold(tmp_path) -> None:
    """旧版 fold 文件不能改：stamp 只写 sidecar，之后指纹检查才真正生效。"""
    cfg = _cfg(tmp_path)
    df = _toy_df()
    _write_processed(cfg, df)
    out = freeze(cfg)

    # 退回旧版状态：meta 无指纹、sidecar 不存在
    meta = json.loads(out.read_text(encoding="utf-8"))
    meta.pop("data_fingerprint")
    out.write_text(json.dumps(meta), encoding="utf-8")
    sidecar = out.with_name(out.stem + ".fingerprint.json")
    sidecar.unlink()
    assert verify_frozen(cfg)["checked"] is False

    before = out.read_bytes()
    written = write_fingerprint(cfg)

    assert written == sidecar and sidecar.exists()
    assert out.read_bytes() == before, "stamp 不能改 fold 文件"
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    for key in ("data_fingerprint", "fold_file_sha256", "n_rows",
                "n_splits", "code_version", "stamped_at"):
        assert key in payload, key

    report = verify_frozen(cfg)
    assert report["checked"] is True
    assert report["ok"] is True
    assert "数据指纹一致" in report["detail"]


def test_verify_frozen_detects_fold_file_edit(tmp_path) -> None:
    """sidecar 里记了 fold 文件 sha256：偷偷改一行 fold_of_row 也会被抓到。"""
    cfg = _cfg(tmp_path)
    _write_processed(cfg, _toy_df())
    out = freeze(cfg)
    meta = json.loads(out.read_text(encoding="utf-8"))
    meta["fold_of_row"] = [(int(f) + 1) % 5 for f in meta["fold_of_row"]]
    out.write_text(json.dumps(meta), encoding="utf-8")

    report = verify_frozen(cfg)
    assert report["checked"] is True
    assert report["ok"] is False
    assert "fold 文件" in report["detail"]


def test_stamp_rejects_data_with_different_row_count(tmp_path) -> None:
    cfg = _cfg(tmp_path)
    _write_processed(cfg, _toy_df())
    out = freeze(cfg)
    out.with_name(out.stem + ".fingerprint.json").unlink()

    _write_processed(cfg, _toy_df(n=1500))
    with pytest.raises(SystemExit):
        write_fingerprint(cfg)
