"""划分策略：冻结 / 读取 / 三方案对比（P1-05）。

铁律：splits/fold_v1.json 一旦冻结就只读。所有实验必须通过 load_folds() 取折，
禁止自己 train_test_split，否则不同人的分数不可比。

为了守住这条铁律：
- freeze() 已存在时默认只提示、不覆盖、退出码 0（`make all` 可重复执行），
  只有显式 --force 才覆盖，且覆盖前自动备份成 .bak.<时间戳>；
- meta 里记录 created_at / data_fingerprint / code_version，
  verify_frozen() 用数据指纹检查"折还是不是这份数据上的折"；
- 旧版 fold 文件（meta 没有指纹）不改 fold_of_row，用 sidecar
  `splits/fold_v1.fingerprint.json` 补指纹：
  `python -m src.common.split stamp --config configs/base.yaml`。
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

from src.common.config import code_version, load_config, path_of, sha256_file
from src.common.config import ROOT


def _load_processed(cfg: dict) -> pd.DataFrame:
    p = path_of(cfg, "processed_dir") / "train.parquet"
    if not p.exists():
        raise SystemExit(f"[split] 找不到 {p}，先跑 make data")
    return pd.read_parquet(p)


def _split_path(cfg: dict | None = None, path: str | Path | None = None) -> Path:
    """冻结文件的绝对路径。"""
    raw = path if path is not None else (cfg or {}).get("paths", {}).get("splits")
    if raw is None:
        raise SystemExit("[split] 没有配置 paths.splits，也没传 path")
    p = Path(raw)
    if not p.is_absolute():
        p = ROOT / p
    return p


def data_fingerprint(cfg: dict) -> dict[str, str]:
    """processed train/test parquet 的 sha256，用来把折叠绑定到具体数据版本。

    parquet 是二进制，sha256_file 不会做换行归一化。
    """
    proc = path_of(cfg, "processed_dir")
    fp: dict[str, str] = {}
    for name in ("train.parquet", "test.parquet"):
        p = proc / name
        if p.exists():
            fp[name] = sha256_file(p)
    return fp


def _fingerprint_sidecar(split_path: Path) -> Path:
    """sidecar 路径：splits/fold_v1.json -> splits/fold_v1.fingerprint.json。"""
    return split_path.with_name(f"{split_path.stem}.fingerprint.json")


def _parquet_num_rows(p: Path) -> int | None:
    """只读 parquet 元数据拿行数，不把 92 MB 数据读进内存。"""
    if not p.exists():
        return None
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(p).metadata.num_rows)
    except Exception:
        return None


def _sidecar_payload(cfg: dict, split_path: Path, meta: dict) -> dict:
    """旁路指纹内容：数据指纹 + fold 文件 sha256 + 形状 + 代码版本。

    fold 文件是 JSON 文本，按 CRLF->LF 归一化后哈希：
    否则 Windows(CRLF) 与 Linux(LF) 检出会算出不同摘要，指纹检查变成平台噪声。
    """
    return {
        "version": meta.get("version", "v1"),
        "stamped_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "fold_file": split_path.name,
        "fold_file_sha256": sha256_file(split_path, normalize_newlines=True),
        "n_rows": len(meta["fold_of_row"]),
        "n_splits": int(meta["n_splits"]),
        "strategy": meta.get("strategy"),
        "data_fingerprint": data_fingerprint(cfg),
        "code_version": code_version(),
    }


def write_fingerprint(cfg: dict, path: str | Path | None = None) -> Path:
    """给已冻结的划分补写旁路指纹（不修改 fold 文件、不动 fold_of_row）。

    用于旧版 fold_v1.json：meta 里没有 data_fingerprint，但文件本身不许改。
    """
    split_path = _split_path(cfg, path)
    _folds, meta = load_folds(cfg, path=split_path)
    n_rows = len(meta["fold_of_row"])

    # 绑定前先确认数据行数没变，否则这份指纹是给"另一份数据"盖的章
    n_data = _parquet_num_rows(path_of(cfg, "processed_dir") / "train.parquet")
    if n_data is not None and n_data != n_rows:
        raise SystemExit(
            f"[split] processed train.parquet 有 {n_data:,} 行，"
            f"冻结划分是 {n_rows:,} 行，拒绝 stamp：\n"
            f"        数据换过或划分过期了，先确认 make data / make split 的顺序。"
        )

    sidecar = _fingerprint_sidecar(split_path)
    payload = _sidecar_payload(cfg, split_path, meta)
    sidecar.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )

    print(f"[split] 指纹旁路 -> {sidecar}")
    print(f"[split] data_fingerprint: {payload['data_fingerprint']}")
    print(f"[split] fold_file_sha256: {payload['fold_file_sha256']}")
    print(f"[split] n_rows={payload['n_rows']:,}  n_splits={payload['n_splits']}  "
          f"code_version={payload['code_version']}")
    print("[split] fold 文件未改动；指纹只写在 sidecar 里。")
    return sidecar


def build_group_array(df: pd.DataFrame, cfg: dict) -> np.ndarray | None:
    """group 策略的实体列。distance_bins = 用 Flight Distance 分箱当伪实体。"""
    spec = cfg["split"].get("group_col")
    if not spec:
        return None
    if spec == "distance_bins":
        col = "Flight Distance"
        if col not in df.columns:
            raise SystemExit(f"[split] 没有 {col}，无法构造 distance_bins")
        n_bins = int(cfg["split"].get("group_bins", 50))
        r = df[col].rank(method="first")
        return pd.qcut(r, n_bins, labels=False).to_numpy()
    if spec not in df.columns:
        raise SystemExit(f"[split] 分组列 {spec} 不存在")
    return df[spec].to_numpy()


def build_folds(df: pd.DataFrame, cfg: dict, strategy: str | None = None) -> np.ndarray:
    """返回 fold_of_row：长度 n 的数组，取值 0..k-1。"""
    y = df[cfg["data"]["target"]].to_numpy()
    n = len(df)
    s = strategy or cfg["split"]["strategy"]
    k = int(cfg["split"]["n_splits"])
    seed = int(cfg["split"]["seed"])
    idx = np.arange(n)
    fold = np.full(n, -1, dtype=np.int16)

    if s == "time":
        col = cfg["split"].get("time_col")
        if not col or col not in df.columns:
            raise SystemExit("[split] time 策略需要 split.time_col 且该列存在")
        order = np.argsort(df[col].to_numpy(), kind="mergesort")
        for f, chunk in enumerate(np.array_split(order, k)):
            fold[chunk] = f
        return fold

    if s == "group":
        g = build_group_array(df, cfg)
        if g is None:
            raise SystemExit("[split] group 策略需要 split.group_col")
        it = GroupKFold(n_splits=k).split(idx, y, g)
    elif s == "kfold":
        it = KFold(n_splits=k, shuffle=True, random_state=seed).split(idx)
    elif s == "stratified":
        it = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed).split(idx, y)
    else:
        raise SystemExit(f"[split] 未知策略: {s}")

    for f, (_, va) in enumerate(it):
        fold[va] = f
    assert (fold >= 0).all(), "有样本没被分到任何折"
    return fold


def load_folds(cfg: dict | None = None, path: str | Path | None = None):
    """读取冻结划分并校验。返回 (folds, meta)，folds 是 [(train_idx, valid_idx), ...]。

    校验项：fold_of_row 存在、是整数的一维数组、长度与 n_rows 一致、
    取值都在 0..k-1、每折非空。坏文件直接 SystemExit，不允许带着错划分训练。
    """
    p = _split_path(cfg, path)
    if not p.exists():
        raise SystemExit(f"[split] 找不到冻结划分 {p}\n        先跑 make split（P1-05 交付物）")

    try:
        meta = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise SystemExit(f"[split] 冻结划分 {p} 不是合法 JSON：{e}")
    if not isinstance(meta, dict):
        raise SystemExit(f"[split] 冻结划分 {p} 顶层应为对象，实际是 {type(meta).__name__}")

    for key in ("n_splits", "fold_of_row"):
        if key not in meta:
            raise SystemExit(f"[split] 冻结划分 {p} 缺少字段 {key!r}")

    try:
        k = int(meta["n_splits"])
    except (TypeError, ValueError):
        raise SystemExit(f"[split] n_splits 不是整数：{meta['n_splits']!r}")
    if k < 2:
        raise SystemExit(f"[split] n_splits={k} 不合法（至少 2 折）")

    raw = meta["fold_of_row"]
    if not isinstance(raw, list) or len(raw) == 0:
        raise SystemExit(f"[split] fold_of_row 应为非空列表，实际是 {type(raw).__name__}")
    arr = np.asarray(raw)
    if arr.ndim != 1:
        raise SystemExit(f"[split] fold_of_row 应为一维，实际 {arr.ndim} 维")
    if np.issubdtype(arr.dtype, np.floating):
        if not np.all(np.mod(arr, 1) == 0):
            raise SystemExit("[split] fold_of_row 含非整数取值")
        arr = arr.astype(np.int64)
    elif not np.issubdtype(arr.dtype, np.integer):
        raise SystemExit(f"[split] fold_of_row dtype={arr.dtype} 不是整数，疑似文件损坏")
    arr = arr.astype(np.int64)

    if "n_rows" in meta:
        try:
            n_rows = int(meta["n_rows"])
        except (TypeError, ValueError):
            raise SystemExit(f"[split] n_rows 不是整数：{meta['n_rows']!r}")
        if n_rows != len(arr):
            raise SystemExit(
                f"[split] fold_of_row 长度 {len(arr)} 与 meta.n_rows={n_rows} 不一致"
            )

    if (arr < 0).any():
        raise SystemExit(f"[split] fold_of_row 含负数 {int((arr < 0).sum())} 个")
    if (arr >= k).any():
        raise SystemExit(
            f"[split] fold_of_row 取值超出 0..{k - 1}（最大值 {int(arr.max())}）"
        )

    counts = np.bincount(arr, minlength=k)
    empty = [f for f in range(k) if counts[f] == 0]
    if empty:
        raise SystemExit(f"[split] 第 {empty} 折为空，划分文件不可用")

    folds = [(np.where(arr != f)[0], np.where(arr == f)[0]) for f in range(k)]
    return folds, meta


def verify_frozen(cfg: dict) -> dict:
    """校验冻结划分与当前 processed 数据 / fold 文件是否仍对应。

    指纹来源优先级：meta.data_fingerprint > sidecar（fold_v1.fingerprint.json）。
    只要 sidecar 在，就额外校验 fold 文件 sha256，能发现"悄悄改了一行 fold_of_row"。
    返回 {"checked": bool, "ok": bool, "detail": str, "path": str}：
    checked=False 表示没有任何指纹可比（旧文件且未 stamp），不判 FAIL。
    """
    p = _split_path(cfg)
    folds, meta = load_folds(cfg)
    n_rows = len(meta["fold_of_row"])
    k = int(meta["n_splits"])
    base = f"{p.name}: {n_rows:,} 行 / {k} 折"
    diffs: list[str] = []

    sidecar_path = _fingerprint_sidecar(p)
    expected = dict(meta.get("data_fingerprint") or {})
    source = "meta" if expected else ""
    fold_sha_expected: str | None = None
    sidecar: dict | None = None
    sidecar_error = ""

    if sidecar_path.exists():
        try:
            loaded = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as e:
            sidecar_error = f"旁路指纹 {sidecar_path.name} 无法解析：{e}"
        else:
            if isinstance(loaded, dict):
                sidecar = loaded
                if not expected:
                    expected = dict(sidecar.get("data_fingerprint") or {})
                    if expected:
                        source = sidecar_path.name
                fold_sha_expected = sidecar.get("fold_file_sha256")
                for key, got in (("n_rows", n_rows), ("n_splits", k)):
                    val = sidecar.get(key)
                    if val is None:
                        continue
                    try:
                        same = int(val) == got
                    except (TypeError, ValueError):
                        diffs.append(f"{sidecar_path.name} 的 {key} 不是整数：{val!r}")
                        continue
                    if not same:
                        diffs.append(
                            f"{sidecar_path.name} 的 {key}={val} 与 fold 文件 {got} 不一致"
                        )
            else:
                sidecar_error = f"旁路指纹 {sidecar_path.name} 顶层不是对象"

    if sidecar_error:
        return {"checked": True, "ok": False, "path": str(p),
                "detail": f"{base}；{sidecar_error}"}

    if not expected:
        detail = (f"{base}；meta 与 {sidecar_path.name} 都没有 data_fingerprint，"
                  f"跳过数据指纹校验（可跑 python -m src.common.split stamp 补写）")
        if diffs:
            detail += "；" + "；".join(diffs)
        return {"checked": False, "ok": not diffs, "path": str(p), "detail": detail}

    current = data_fingerprint(cfg)
    for name, digest in sorted(expected.items()):
        got = current.get(name)
        if got is None:
            diffs.append(f"{name} 不存在")
        elif got != digest:
            diffs.append(f"{name} 指纹变化")
    if fold_sha_expected:
        got = sha256_file(p, normalize_newlines=True)
        if got != fold_sha_expected:
            diffs.append("fold 文件本身与 stamped 的 sha256 不一致（划分被改过）")
    if not folds or not all(len(va) > 0 for _, va in folds):
        diffs.append("存在空折")

    code = meta.get("code_version") or (sidecar or {}).get("code_version") or "未记录"
    detail = f"{base}；指纹来源={source}"
    if fold_sha_expected:
        detail += "（含 fold 文件 sha256）"
    detail += f"；code_version={code}"
    detail += "；" + ("；".join(diffs) if diffs else
                     f"数据指纹一致（{', '.join(sorted(expected))}）")
    return {"checked": True, "ok": not diffs, "path": str(p), "detail": detail}


def _backup_path(out: Path) -> Path:
    """fold_v1.json.bak.<YYYYmmdd-HHMMSS>，同一秒内多次覆盖则加序号。"""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    bak = out.with_name(f"{out.name}.bak.{stamp}")
    i = 1
    while bak.exists():
        bak = out.with_name(f"{out.name}.bak.{stamp}-{i}")
        i += 1
    return bak


def freeze(cfg: dict, force: bool = False) -> Path | None:
    """冻结划分。

    文件已存在且未指定 force：打印提示、原样保留、返回 None（CLI 退出码 0），
    这样 `make all` 可以反复执行而不会误覆盖历史划分。
    指定 force：先把旧文件备份成 `.bak.<时间戳>`，再写新文件。
    """
    out = _split_path(cfg)
    if out.exists() and not force:
        print(f"[split] 已存在冻结划分 {out}，本次跳过（未指定 --force）。")
        print("[split] fold_v1.json 是只读契约：历史上所有实验、报告里的分数都引用它，")
        print("[split] 覆盖会让过去的结论不可复现，所以默认不重写。")
        try:
            fr = verify_frozen(cfg)
        except SystemExit as e:
            print(f"[split][警告] 现有划分无法校验：{e}")
        else:
            mark = "" if fr["ok"] else "[警告] "
            print(f"[split] {mark}{fr['detail']}")
            if fr["checked"] and not fr["ok"]:
                print("[split][警告] 折叠与当前 processed 数据不匹配：确认数据是否被换过")
        print(f"[split] 确实要重新划分时加 --force（旧文件会先备份为 {out.name}.bak.<时间戳>）。")
        return None

    df = _load_processed(cfg)
    fold = build_folds(df, cfg)
    y = df[cfg["data"]["target"]].to_numpy()
    k = int(cfg["split"]["n_splits"])

    out.parent.mkdir(parents=True, exist_ok=True)
    backup = None
    if out.exists():
        backup = _backup_path(out)
        shutil.copy2(out, backup)
        print(f"[split] 旧划分已备份 -> {backup}")

    meta = {
        "version": "v1",
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "strategy": cfg["split"]["strategy"],
        "n_splits": k,
        "seed": int(cfg["split"]["seed"]),
        "target": cfg["data"]["target"],
        "group_col": cfg["split"].get("group_col"),
        "n_rows": len(df),
        "data_fingerprint": data_fingerprint(cfg),
        "code_version": code_version(),
        "fold_counts": [int((fold == f).sum()) for f in range(k)],
        "positive_rate_by_fold": [round(float(y[fold == f].mean()), 6) for f in range(k)],
        "fold_of_row": fold.tolist(),
    }
    out.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8", newline="\n")

    print(f"[split] 策略={meta['strategy']} 折数={k} seed={meta['seed']} 行数={meta['n_rows']:,}")
    print(f"[split] 每折正例率: {meta['positive_rate_by_fold']}")
    print(f"[split] 数据指纹: {meta['data_fingerprint']}")
    print(f"[split] 代码版本: {meta['code_version']}")
    print(f"[split] 已冻结 -> {out}")
    # 同时写旁路指纹：verify 即便只读 fold 文件，也能校验"数据 + 文件"两件事
    sidecar = _fingerprint_sidecar(out)
    sidecar.write_text(
        json.dumps(_sidecar_payload(cfg, out, meta), ensure_ascii=False, indent=2),
        encoding="utf-8",
        newline="\n",
    )
    print(f"[split] 指纹旁路 -> {sidecar}")
    print("[split] 从现在起该文件只读；任何人改它，其历史实验作废。")
    return out


def compare(cfg: dict, score: bool = False) -> None:
    df = _load_processed(cfg)
    y = df[cfg["data"]["target"]].to_numpy()
    k = int(cfg["split"]["n_splits"])
    rows = []

    for s in ("kfold", "stratified", "group"):
        try:
            fold = build_folds(df, cfg, strategy=s)
        except SystemExit as e:
            print(f"[compare] 跳过 {s}: {e}")
            continue
        rates = [float(y[fold == f].mean()) for f in range(k)]
        sizes = [int((fold == f).sum()) for f in range(k)]
        row = {
            "strategy": s,
            "n_splits": k,
            "pos_mean": round(float(np.mean(rates)), 6),
            "pos_std": round(float(np.std(rates)), 6),
            "fold_size_min": min(sizes),
            "fold_size_max": max(sizes),
        }
        if score:
            from src.models.train import run_cv

            res = run_cv(cfg, model="lgbm", groups=["base"], fold_array=fold,
                         log=False, verbose=False)
            row["cv_mean"] = round(res["cv_mean"], 6)
            row["cv_std"] = round(res["cv_std"], 6)
        rows.append(row)

    table = pd.DataFrame(rows)
    print("\n[compare] 三方案对比")
    print(table.to_string(index=False))

    out = Path(cfg["paths"]["reports"]) / f"split_compare_{cfg['project']}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n[compare] 已写入 {out}")
    print("[compare] 选型口径：正例率波动小 + CV 与 LB 差距最小；不要只挑分数最高。")


def main() -> None:
    ap = argparse.ArgumentParser(description="划分策略管理")
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("freeze", help="冻结 fold_v1.json")
    f.add_argument("--config", default="configs/base.yaml")
    f.add_argument(
        "--force",
        action="store_true",
        help="覆盖已冻结的划分（会先把旧文件备份为 .bak.<时间戳>）",
    )

    c = sub.add_parser("compare", help="三方案对比")
    c.add_argument("--config", default="configs/base.yaml")
    c.add_argument("--score", action="store_true", help="同时跑 lgbm base 打分")

    s = sub.add_parser(
        "stamp",
        help="为已冻结的划分补写指纹旁路文件（不改 fold_of_row）",
    )
    s.add_argument("--config", default="configs/base.yaml")

    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.cmd == "freeze":
        freeze(cfg, force=args.force)
    elif args.cmd == "stamp":
        write_fingerprint(cfg)
    else:
        compare(cfg, score=args.score)


if __name__ == "__main__":
    main()
