"""evaluate / badcase / stability / _io 的单元与集成测试。

全部使用 tmp_path 下的 200 行合成数据，不读 70 万真实集、不跑 LightGBM。
集成测试会真正生成 metrics.json / 图 / summary，验证文件与字段存在。
"""

from __future__ import annotations

import json
import math
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from src.eval import _io, badcase, evaluate, stability


def _toy(seed: int = 3, n: int = 200) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.45).astype("int64")
    logit = np.where(y == 1, 1.4, -1.4) + rng.normal(0.0, 0.9, size=n)
    pred = 1.0 / (1.0 + np.exp(-logit))
    df = pd.DataFrame(
        {
            "id": np.arange(10_000, 10_000 + n),
            "Age": rng.integers(18, 70, n),
            "Flight Distance": rng.integers(100, 4000, n),
            "Online boarding": rng.integers(0, 6, n),
            "Departure Delay in Minutes": rng.integers(0, 120, n),
            "Class": rng.choice(["Business", "Eco", "Eco Plus"], n, p=[0.5, 0.4, 0.1]),
            "Type of Travel": rng.choice(["Business travel", "Personal Travel"], n),
            "Customer Type": rng.choice(["Loyal Customer", "disloyal Customer"], n),
            "satisfaction": y,
        }
    )
    return df, y, pred


def _write_dataset(
    tmp_path, df: pd.DataFrame, y: np.ndarray, pred: np.ndarray
) -> tuple[dict, object]:
    processed = tmp_path / "processed"
    processed.mkdir(exist_ok=True)
    frame = df.copy()
    frame["satisfaction"] = y
    frame.to_parquet(processed / "train.parquet", index=False)

    oof = pd.DataFrame(
        {"id": frame["id"].to_numpy(), "row": np.arange(len(frame)), "y": y, "pred": pred}
    )
    oof_path = tmp_path / "oof.parquet"
    oof.to_parquet(oof_path, index=False)

    cfg = {
        "project": "toy",
        "seed": 42,
        "_path": "configs/toy-test.yaml",
        "paths": {
            "processed_dir": str(processed),
            "oof_dir": str(tmp_path / "oof"),
            "reports": str(tmp_path / "reports"),
            "exp_log": str(tmp_path / "exp_log.csv"),
            "splits": "splits/fold_v1.json",
        },
        "data": {"target": "satisfaction", "id_col": "id"},
        "features": {"service_cols": [], "cat_cols": [], "num_cols": []},
        "split": {"n_splits": 5},
        "model": {"lgbm": {}},
    }
    return cfg, oof_path


# ---------------------------------------------------------------- _io / OOF


def test_load_oof_sorts_by_row_and_keeps_alignment(tmp_path) -> None:
    p = tmp_path / "oof.parquet"
    pd.DataFrame(
        {"row": [2, 0, 1], "y": [1, 0, 1], "pred": [0.9, 0.1, 0.8]}
    ).to_parquet(p, index=False)

    out = _io.load_oof(p, n_rows=3)

    assert out["row"].tolist() == [0, 1, 2]
    assert out["y"].tolist() == [0, 1, 1]
    assert out["pred"].tolist() == pytest.approx([0.1, 0.8, 0.9])


def test_load_oof_missing_column_raises(tmp_path) -> None:
    p = tmp_path / "oof.parquet"
    pd.DataFrame({"row": [0, 1], "y": [0, 1]}).to_parquet(p, index=False)

    with pytest.raises(ValueError, match="缺少列"):
        _io.load_oof(p, n_rows=2)


def test_load_oof_length_mismatch_and_allow_partial(tmp_path) -> None:
    p = tmp_path / "oof.parquet"
    pd.DataFrame(
        {"row": np.arange(100), "y": np.zeros(100, dtype=int), "pred": np.full(100, 0.5)}
    ).to_parquet(p, index=False)

    with pytest.raises(ValueError, match="allow-partial"):
        _io.load_oof(p, n_rows=200)

    with pytest.warns(UserWarning, match="仅覆盖"):
        out = _io.load_oof(p, n_rows=200, allow_partial=True)
    assert len(out) == 100


def test_load_oof_nan_pred_raises(tmp_path) -> None:
    p = tmp_path / "oof.parquet"
    pd.DataFrame({"row": [0, 1], "y": [0, 1], "pred": [0.2, np.nan]}).to_parquet(
        p, index=False
    )

    with pytest.raises(ValueError, match="缺失"):
        _io.load_oof(p, n_rows=2)


def test_load_oof_duplicate_row_raises(tmp_path) -> None:
    p = tmp_path / "oof.parquet"
    pd.DataFrame({"row": [0, 0, 1], "y": [0, 1, 1], "pred": [0.2, 0.3, 0.9]}).to_parquet(
        p, index=False
    )

    with pytest.raises(ValueError, match="重复"):
        _io.load_oof(p, n_rows=3)


def test_load_oof_clips_out_of_range(tmp_path) -> None:
    p = tmp_path / "oof.parquet"
    pd.DataFrame({"row": [0, 1], "y": [0, 1], "pred": [-0.2, 1.2]}).to_parquet(
        p, index=False
    )

    with pytest.warns(UserWarning, match="越界"):
        out = _io.load_oof(p, n_rows=2)

    assert out["pred"].tolist() == [0.0, 1.0]


def test_resolve_oof_falls_back_to_latest_run(tmp_path, capsys) -> None:
    oof_dir = tmp_path / "oof"
    oof_dir.mkdir()
    p = oof_dir / "P2_feat_lgbm_CT_1030_main.parquet"
    pd.DataFrame({"row": [0], "y": [0], "pred": [0.5]}).to_parquet(p, index=False)
    cfg = {"paths": {"oof_dir": str(oof_dir)}}

    found = _io.resolve_oof_path(cfg)

    assert found == p
    assert "最新 run" in capsys.readouterr().out


def test_resolve_oof_missing_raises_actionable(tmp_path) -> None:
    cfg = {"paths": {"oof_dir": str(tmp_path / "none")}}

    with pytest.raises(FileNotFoundError, match="repository|OOF|python -m"):
        _io.resolve_oof_path(cfg)


# ---------------------------------------------------------------- 分群工具


def test_delay_bucket_boundaries() -> None:
    s = pd.Series([0, 1, 15, 16, 60, 61, np.nan, -5], dtype="float64")
    out = _io.delay_bucket(s).tolist()

    assert out == ["0", "1-15", "1-15", "16-60", "16-60", ">60", "缺失", "0"]


def test_distance_quintile_counts_and_missing() -> None:
    s = pd.Series(np.arange(100, dtype="float64"))
    q = _io.distance_quintile(s)
    counts = q.value_counts().to_dict()

    for label in ("Q1", "Q2", "Q3", "Q4", "Q5"):
        assert counts[label] == 20

    q2 = _io.distance_quintile(pd.Series([1.0, np.nan, 2.0, 3.0, 4.0, 5.0]))
    assert "缺失" in q2.tolist()


def test_build_segment_frame_columns() -> None:
    df, _, _ = _toy()
    seg = _io.build_segment_frame(df)

    assert list(seg.columns) == _io.SEGMENT_COLUMNS
    assert len(seg) == len(df)
    assert set(seg["Class"].unique()) == {"Business", "Eco", "Eco Plus"}


def test_ensure_lgbm_dtypes_converts_str_only() -> None:
    X = pd.DataFrame(
        {"s": pd.Series(["a", "b", "a"], dtype="str"), "n": [1, 2, 3]}
    )
    out = _io.ensure_lgbm_dtypes(X, verbose=False)

    assert isinstance(out["s"].dtype, pd.CategoricalDtype)
    assert pd.api.types.is_numeric_dtype(out["n"].dtype)


def test_segment_metrics_marks_low_n_and_single_class_auc_nan() -> None:
    df, y, pred = _toy()
    feats = df.copy()
    feats.loc[:9, "Class"] = "Rare"
    y = y.copy()
    y[:10] = 0
    pred = pred.copy()
    pred[:10] = 0.1

    table = evaluate.segment_metrics(y, pred, feats, threshold=0.5, min_n=50)

    required = {"segment", "group", "n", "pos_rate", "auc", "pred_mean", "logloss", "low_n"}
    assert required <= set(table.columns)

    rare = table[(table["segment"] == "Class") & (table["group"] == "Rare")].iloc[0]
    assert bool(rare["low_n"]) is True
    assert math.isnan(float(rare["auc"]))

    business = table[(table["segment"] == "Class") & (table["group"] == "Business")].iloc[0]
    assert bool(business["low_n"]) is False
    assert float(business["auc"]) > 0.5


# ---------------------------------------------------------------- evaluate 集成


def test_run_evaluation_outputs(tmp_path) -> None:
    df, y, pred = _toy()
    cfg, oof_path = _write_dataset(tmp_path, df, y, pred)
    out = tmp_path / "eval"

    res = evaluate.run_evaluation(
        cfg, oof_path=oof_path, out_dir=out, n_boot=20, n_bins=5
    )

    for name in (
        "metrics.json", "reliability.csv", "segment_metrics.csv", "summary.md",
        "p3_calibration.png", "p3_score_dist.png", "p3_confusion.png",
    ):
        path = out / name
        assert path.exists(), name
        assert path.stat().st_size > 0, name

    data = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert data["n"] == 200
    assert data["auc"] > 0.8
    assert data["auc_ci"]["n_boot"] == 20
    assert data["segment_summary"]["n_groups"] > 0
    assert res["metrics"]["auc"] == pytest.approx(data["auc"])

    seg = pd.read_csv(out / "segment_metrics.csv")
    assert {"segment", "group", "n", "pos_rate", "auc", "pred_mean", "logloss", "low_n"} <= set(
        seg.columns
    )
    assert bool(seg["low_n"].any()) is True

    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "ROC AUC" in summary
    assert "low_n" in summary


# ---------------------------------------------------------------- badcase


def test_load_badcases_fp_fn_top_and_columns() -> None:
    df, y, pred = _toy()
    y = y.copy()
    pred = pred.copy()
    y[:5] = 1
    pred[:5] = 0.05
    y[5:10] = 0
    pred[5:10] = 0.95
    oof = pd.DataFrame({"row": np.arange(len(df)), "y": y, "pred": pred})

    bad = badcase.load_badcases(oof, df, fp_cut=0.8, fn_cut=0.2, top=50)

    assert list(bad.columns) == badcase.OUT_COLUMNS
    assert len(bad) == 10
    assert int((bad["error_type"] == "FP").sum()) == 5
    assert int((bad["error_type"] == "FN").sum()) == 5
    fp = bad[bad["error_type"] == "FP"]
    fn = bad[bad["error_type"] == "FN"]
    assert fp["pred"].tolist() == sorted(fp["pred"].tolist(), reverse=True)
    assert fn["pred"].tolist() == sorted(fn["pred"].tolist())
    assert fp["y"].eq(0).all() and fn["y"].eq(1).all()


def test_segment_error_table_marks_low_n() -> None:
    df, y, pred = _toy()
    feats = df.copy()
    feats.loc[:9, "Class"] = "Rare"

    table = badcase.segment_error_table(y, pred, feats, threshold=0.5, min_n=50)

    assert {"segment", "group", "n", "n_error", "error_rate", "fp", "fn", "low_n"} <= set(
        table.columns
    )
    assert bool(table[(table["segment"] == "Class") & (table["group"] == "Rare")]["low_n"].iloc[0]) is True
    assert table["error_rate"].between(0.0, 1.0).all()


def test_run_badcase_outputs(tmp_path) -> None:
    df, y, pred = _toy()
    y = y.copy()
    pred = pred.copy()
    y[:5] = 1
    pred[:5] = 0.05
    y[5:10] = 0
    pred[5:10] = 0.95
    cfg, oof_path = _write_dataset(tmp_path, df, y, pred)
    out = tmp_path / "badcase"

    res = badcase.run_badcase(cfg, oof_path=oof_path, out_dir=out, top=50)

    assert (out / "badcase_top50.csv").exists()
    assert (out / "badcase_summary.md").exists()
    assert (out / "p3_badcase.png").exists()
    csv = pd.read_csv(out / "badcase_top50.csv")
    assert list(csv.columns) == badcase.OUT_COLUMNS
    assert len(csv) == 10
    assert len(res["bad"] ) == 10


# ---------------------------------------------------------------- stability


def test_parse_seeds_dedupes_and_errors() -> None:
    assert stability.parse_seeds("42, 2026,42") == [42, 2026]
    with pytest.raises(ValueError, match="不是整数"):
        stability.parse_seeds("42,abc")
    with pytest.raises(ValueError, match="为空"):
        stability.parse_seeds(" , ")


def test_summarize_runs() -> None:
    runs = [
        {"seed": 42, "cv_mean": 0.9600, "cv_std": 0.0010},
        {"seed": 7, "cv_mean": 0.9620, "cv_std": 0.0008},
    ]

    summary = stability.summarize_runs(runs, baseline=0.9600)

    assert summary["n_seeds"] == 2
    assert summary["seed_mean_auc"] == pytest.approx(0.9610)
    assert summary["seed_std_auc"] == pytest.approx(0.00141421356, rel=1e-6)
    assert summary["seed_range"] == pytest.approx(0.0020)
    assert summary["mean_fold_std"] == pytest.approx(0.0009)
    assert summary["mean_delta_vs_baseline"] == pytest.approx(0.0010)


def test_plot_seed_variance_single_seed(tmp_path) -> None:
    runs = [
        {
            "seed": 42,
            "run_id": "P3_eval_lgbm_CT_1031_s42_stab",
            "cv_mean": 0.960191,
            "cv_std": 0.000583,
            "fold_aucs": [0.9603, 0.9591, 0.9609, 0.9604, 0.9602],
        }
    ]
    summary = stability.summarize_runs(runs, baseline=0.960191)
    path = tmp_path / "p3_seed_variance.png"

    stability._plot_seed_variance(runs, summary, 0.960191, path)

    assert path.exists() and path.stat().st_size > 0


def test_stability_dry_run_does_not_write(tmp_path, monkeypatch, capsys) -> None:
    splits = tmp_path / "fold.json"
    splits.write_text("{}", encoding="utf-8")
    exp_log = tmp_path / "exp_log.csv"
    exp_log.write_text(
        "run_id,model,feat_ver,cv_mean\nP2_feat_lgbm_CT_1030_main,lgbm,C+T,0.960191\n",
        encoding="utf-8",
    )
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "project": "toy",
                "seed": 42,
                "paths": {"splits": str(splits), "exp_log": str(exp_log)},
                "split": {"n_splits": 5},
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    out = tmp_path / "eval"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "stability",
            "--config", str(cfg_path),
            "--seeds", "42,7",
            "--dry-run",
            "--out", str(out),
        ],
    )

    stability.main()
    printed = capsys.readouterr().out

    assert "DRY-RUN" in printed
    assert "2 × 5 = 10" in printed
    assert not out.exists()
