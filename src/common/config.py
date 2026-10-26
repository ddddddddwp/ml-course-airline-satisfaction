"""配置加载与路径工具。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = "configs/base.yaml") -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise SystemExit(f"[配置] 找不到 {p}")
    with p.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_path"] = str(p)
    return cfg


def path_of(cfg: dict, key: str) -> Path:
    """取配置里的路径，统一转成绝对路径并确保父目录存在。"""
    p = Path(cfg["paths"][key])
    if not p.is_absolute():
        p = ROOT / p
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_dir(p: str | Path) -> Path:
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def params_hash(d: dict) -> str:
    s = json.dumps(d, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:8]


def sha256_file(p: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_commit() -> str:
    """取当前 commit 短哈希；不在 git 仓库里时返回空串。"""
    import subprocess

    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""
