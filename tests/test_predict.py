"""predict_test 的契约测试：run_id 含 seed、写 exp_log/OOF、与 run_cv 同协议。

全部用 monkeypatch 注入小数据（120 行），不真跑 70 万行：
- load_folds 被替换成固定 2 折；processed 数据写到 tmp_path。
- 特征构造仍走真实的 FeatureBuilder / LightGBM，不 mock 掉被测逻辑。
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.models import predict as predict_mod
from src.models import train as train_mod
from src.models.train import make_run_id

SERVICE = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Cleanliness",
]
CAT_COLS = ["Gender", "Customer Type", "Type of Travel", "Class"]
NUM_COLS = ["Age", "Flight Distance", "Departure Delay in Minutes", "Arrival Delay in Minutes"]
N_TRAIN, N_TEST, N_FOLDS = 400, 60, 2


def _make_frame(n: int, seed: int, with_target: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    data: dict[str, object] = {"id": np.arange(1000, 1000 + n)}
    data["Age"] = rng.integers(18, 70, n)
    data["Flight Distance"] = rng.integers(80, 5000, n)
    for c in SERVICE:
        data[c] = rng.integers(0, 6, n)
    data["Departure Delay in Minutes"] = rng.integers(0, 200, n)
    arr = rng.integers(0, 200, n).astype("float64")
    arr[::13] = np.nan
    data["Arrival Delay in Minutes"] = arr
    data["arr_delay_missing"] = np.isnan(arr).astype("int8")
    data["Gender"] = rng.choice(["Female", "Male"], n)
    data["Customer Type"] = rng.choice(["Loyal Customer", "disloyal Customer"], n)
    data["Type of Travel"] = rng.choice(["Business travel", "Personal Travel"], n)
    data["Class"] = rng.choice(["Business", "Eco", "Eco Plus"], n)
    if with_target:
        data["satisfaction"] = (rng.random(n) < 0.45).astype("int8")
    return pd.DataFrame(data)


def _folds(n: int = N_TRAIN, k: int = N_FOLDS):
    fold = np.arange(n) % k
    return [(np.where(fold != f)[0], np.where(fold == f)[0]) for f in range(k)]


def _prep(tmp_path: Path) -> dict:
    """写一份小型 processed 数据 + 指向 tmp 的 cfg。"""
    proc = tmp_path / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    _make_frame(N_TRAIN, seed=0).to_parquet(proc / "train.parquet", index=False)
    _make_frame(N_TEST, seed=1, with_target=False).to_parquet(proc / "test.parquet", index=False)
    return {
        "seed": 42,
        "paths": {
            "processed_dir": str(proc),
            "reports": str(tmp_path / "reports"),
            "figures": str(tmp_path / "reports" / "figures"),
            "oof_dir": str(tmp_path / "reports" / "oof"),
            "exp_log": str(tmp_path / "reports" / "exp_log.csv"),
            "splits": str(tmp_path / "splits" / "fold.json"),
        },
        "data": {"target": "satisfaction", "id_col": "id"},
        "features": {"service_cols": SERVICE, "cat_cols": CAT_COLS, "num_cols": NUM_COLS},
        "model": {
            "lgbm": {
                "n_estimators": 20, "learning_rate": 0.1, "num_leaves": 15,
                "min_child_samples": 5, "verbose": -1, "n_jobs": 1,
            },
        },
    }


@pytest.fixture
def patched(tmp_path, monkeypatch):
    cfg = _prep(tmp_path)
    folds = _folds()
    monkeypatch.setattr(predict_mod, "load_folds", lambda cfg=None: (folds, {"version": "v1-test"}))
    monkeypatch.setattr(train_mod, "load_folds", lambda cfg=None: (folds, {"version": "v1-test"}))
    return cfg


# --------------------------------------------------------------------------- run_id

def test_make_run_id_contains_seed_and_differs():
    a = make_run_id(owner="P1", stage="feat", model="lgbm", feat_ver="base", seed=42, tag="abl")
    b = make_run_id(owner="P1", stage="feat", model="lgbm", feat_ver="base", seed=7, tag="abl")

    assert re.fullmatch(r"P1_feat_lgbm_base_\d{4}_s42_abl", a), a
    assert a.endswith("_s42_abl") and b.endswith("_s7_abl")
    assert a != b   # 多 seed 不再互相覆盖 OOF


def test_make_run_id_tags_and_feat_versions():
    rid = make_run_id(owner="P2", stage="final", model="lgbm", feat_ver="C+T", seed=7, tag="final")
    assert re.fullmatch(r"P2_final_lgbm_CT_\d{4}_s7_final", rid), rid
    # 无 tag 时不以 "_" 结尾
    assert not make_run_id(seed=42).endswith("_")


# --------------------------------------------------------------------------- predict_test

def test_predict_test_writes_log_oof_and_submission(patched):
    res = predict_mod.predict_test(patched, groups=["base", "C", "T"], seed=42, verbose=False)

    assert re.fullmatch(r"P2_final_lgbm_CT_\d{4}_s42_final", res["run_id"]), res["run_id"]
    assert res["test_prob"].shape == (N_TEST,)
    assert np.isfinite(res["test_prob"]).all()
    assert 0.0 <= res["cv_mean"] <= 1.0
    assert len(res["oof"]) == N_TRAIN and not np.isnan(res["oof"]).any()

    # exp_log：stage=final，指标/hash/commit 齐全
    log = pd.read_csv(patched["paths"]["exp_log"])
    assert len(log) == 1
    row = log.iloc[0]
    assert row["run_id"] == res["run_id"]
    assert row["stage"] == "final"
    assert row["owner"] == "P2"
    assert row["model"] == "lgbm"
    assert row["feat_ver"] == "C+T"
    assert row["split_ver"] == "v1-test"
    assert len(str(row["params_hash"])) == 8
    assert float(row["cv_mean"]) == pytest.approx(round(res["cv_mean"], 6))
    assert float(row["cv_std"]) == pytest.approx(round(res["cv_std"], 6))
    assert re.match(r"n_feat=\d+", str(row["notes"]))
    assert "git_commit" in log.columns

    # OOF：run_id 命名 + 旧路径兼容，带 id 列
    oof_dir = Path(patched["paths"]["oof_dir"])
    new_oof = oof_dir / f"{res['run_id']}.parquet"
    legacy = oof_dir / "P2_final_oof.parquet"
    assert new_oof.exists() and legacy.exists()
    o = pd.read_parquet(new_oof)
    assert {"id", "row", "y", "pred"} <= set(o.columns)
    assert len(o) == N_TRAIN
    pd.testing.assert_frame_equal(o, pd.read_parquet(legacy))
    np.testing.assert_allclose(o["pred"].to_numpy(), res["oof"], rtol=0, atol=1e-12)

    # submission：概率而非 0/1
    sub = pd.read_csv(Path(patched["paths"]["reports"]) / "submission_final.csv")
    assert list(sub.columns) == ["id", "satisfaction"]
    assert len(sub) == N_TEST
    assert sub["satisfaction"].between(0.0, 1.0).all()
    assert sub["satisfaction"].nunique() > 2


def test_predict_test_different_seed_different_run_id(patched):
    r1 = predict_mod.predict_test(patched, groups=["base", "C", "T"], seed=42, verbose=False)
    r2 = predict_mod.predict_test(patched, groups=["base", "C", "T"], seed=7, verbose=False)

    assert r1["run_id"] != r2["run_id"]
    assert r1["run_id"].endswith("_s42_final")
    assert r2["run_id"].endswith("_s7_final")

    log = pd.read_csv(patched["paths"]["exp_log"])
    assert len(log) == 2
    assert log["run_id"].is_unique


def test_predict_test_no_log(patched):
    res = predict_mod.predict_test(patched, groups=["base", "C", "T"], verbose=False, log=False)
    assert not Path(patched["paths"]["exp_log"]).exists()
    # 但交付物照样写
    assert (Path(patched["paths"]["oof_dir"]) / f"{res['run_id']}.parquet").exists()
    assert (Path(patched["paths"]["reports"]) / "submission_final.csv").exists()


def test_predict_matches_run_cv_on_same_folds(patched):
    """同折、同 seed、同特征组时，predict 的 OOF 必须与 run_cv 逐位一致。

    这是 B 组 dist_bin 修复的端到端证据：predict 逐折 prepare_X 而 run_cv 全量算一次，
    只要边界固定在全量 train 上，两边就是同一套特征。
    """
    groups = ["base", "B", "C", "T"]
    cv = train_mod.run_cv(patched, model="lgbm", groups=groups, seed=42,
                          log=False, verbose=False)
    pred = predict_mod.predict_test(patched, groups=groups, seed=42, verbose=False, log=False)

    np.testing.assert_allclose(pred["oof"], cv["oof"], rtol=0, atol=1e-12)
    assert pred["cv_mean"] == pytest.approx(cv["cv_mean"], rel=0, abs=1e-12)


def test_train_final_docstring_warns_about_shap():
    doc = predict_mod.train_final.__doc__ or ""
    assert "SHAP" in doc and "OOF" in doc
    assert "src/eval/interpret.py" in doc
