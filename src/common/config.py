"""配置加载与路径工具。"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

# 这些后缀按文本处理：哈希前把 CRLF 归一化成 LF。
# 原因：core.autocrlf=true 的 Windows 工作区拿到的是 CRLF，而库里提交的
# CHECKSUMS.txt 是按 LF 算的；不归一化的话 `sha256sum -c` 在 Windows 必然失败。
TEXT_SUFFIXES = {".csv", ".txt"}


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


def sha256_file(
    p: str | Path,
    chunk: int = 1 << 20,
    normalize_newlines: bool | None = None,
) -> str:
    """文件 sha256。

    normalize_newlines=None（默认）：.csv/.txt 先把 CRLF 归一化成 LF 再哈希；
    二进制文件（parquet/png 等）或显式传 False 时逐字节哈希。

    这样同一份 CSV 在 Windows（CRLF 工作区）与 Linux/CI（LF）得到同一个
    摘要，CHECKSUMS.txt 才是跨平台可用的凭据。归一化按块流式做，65 MB 的
    train.csv 不需要整个读进内存，且不会漏掉跨块边界的 "\\r\\n"。
    """
    path = Path(p)
    if normalize_newlines is None:
        normalize_newlines = path.suffix.lower() in TEXT_SUFFIXES

    h = hashlib.sha256()
    with open(path, "rb") as f:
        carry = b""
        while True:
            block = f.read(chunk)
            if not block:
                break
            if normalize_newlines:
                block = carry + block
                carry = b""
                # 块尾可能是 "\r"，下一块开头若是 "\n" 才是 CRLF，先扣住
                if block.endswith(b"\r"):
                    block, carry = block[:-1], b"\r"
                block = block.replace(b"\r\n", b"\n")
            h.update(block)
        if carry:
            h.update(carry)
    return h.hexdigest()


# ---------------------------------------------------------------- 校验和


def read_checksums(p: str | Path) -> dict[str, str]:
    """读 `sha256sum` 格式的校验和文件：<64 位 hex>  <文件名>。

    容忍 CRLF/空行/`#` 注释，以及二进制模式下 sha256sum 输出在文件名前
    加的 `*`。格式不对直接报错，避免"看起来校验过了"的假象。
    """
    path = Path(p)
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if len(parts) != 2:
            raise SystemExit(
                f"[校验] {path}:{lineno} 格式不对，应为 '<sha256>  <文件名>'：{line!r}"
            )
        digest, name = parts[0].lower(), parts[1].strip()
        if name.startswith("*"):
            name = name[1:]
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise SystemExit(f"[校验] {path}:{lineno} 不是 sha256 摘要：{digest!r}")
        out[name] = digest
    return out


def write_checksums(p: str | Path, files) -> dict[str, str]:
    """按 `sha256sum` 格式写校验和文件（固定 LF，Windows 上也能被 -c 读）。"""
    path = Path(p)
    paths = [Path(f) for f in files]
    entries = {f.name: sha256_file(f) for f in paths}
    path.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" 关掉 Windows 的 \n -> \r\n 转换：sha256sum -c 不认带回车的行尾
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for name, digest in entries.items():
            f.write(f"{digest}  {name}\n")
    return entries


def verify_checksums(raw_dir: str | Path, files, path: str | Path | None = None) -> dict:
    """把当前文件与 CHECKSUMS.txt 比对（CSV 按 LF 归一化后哈希）。

    返回 dict：
      exists        CHECKSUMS.txt 是否存在
      verified      摘要一致的条目
      mismatched    摘要不一致的条目（数据可能被替换）
      missing      当前文件在 CHECKSUMS.txt 里没有记录
      absent        有记录但文件不存在
      extra         CHECKSUMS.txt 里多出来、本次没校验的条目
      ok            没有任何问题
    """
    raw = Path(raw_dir)
    ck_path = Path(path) if path is not None else raw / "CHECKSUMS.txt"
    expected = read_checksums(ck_path)
    report: dict = {
        "path": str(ck_path),
        "exists": ck_path.exists(),
        "verified": [],
        "mismatched": [],
        "missing": [],
        "absent": [],
        "extra": [],
        "ok": False,
    }
    names: set[str] = set()
    for f in files:
        fp = Path(f)
        # 兼容三种写法：绝对路径 / 相对仓库根（如 data/raw/train.csv）/ 纯文件名。
        # 纯文件名在 cwd 下通常不存在，再回落到 raw_dir。
        if not fp.is_absolute() and not fp.exists():
            fp = raw / fp
        name = fp.name
        names.add(name)
        if name not in expected:
            report["missing"].append(name)
        elif not fp.exists():
            report["absent"].append(name)
        elif sha256_file(fp) == expected[name]:
            report["verified"].append(name)
        else:
            report["mismatched"].append(name)

    report["extra"] = sorted(set(expected) - names)
    report["ok"] = bool(
        report["exists"]
        and report["verified"]
        and not report["mismatched"]
        and not report["missing"]
        and not report["absent"]
    )
    return report


def checksum_problems(report: dict) -> list[str]:
    """把 verify_checksums 的结果翻译成给人看的毛病列表（空列表 = 没问题）。"""
    problems: list[str] = []
    if not report["exists"]:
        problems.append(f"{report['path']} 不存在")
    for name in report["mismatched"]:
        problems.append(f"{name} 摘要与 CHECKSUMS.txt 不一致")
    for name in report["missing"]:
        problems.append(f"{name} 没有记录在 CHECKSUMS.txt 里")
    for name in report["absent"]:
        problems.append(f"{name} 有记录但文件不存在")
    return problems


# ---------------------------------------------------------------- 版本信息


def git_commit() -> str:
    """当前 commit 短哈希。

    优先用 GIT_COMMIT 环境变量（CI / 没有 .git 的环境），取不到再问 git；
    都取不到返回 "unknown"——空串会让人误以为这个字段忘了写。
    """
    env = (os.environ.get("GIT_COMMIT") or "").strip()
    if env:
        return env
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=5,
        )
        commit = out.stdout.strip()
        if out.returncode == 0 and commit:
            return commit
    except Exception:
        pass
    return "unknown"


def git_dirty() -> bool:
    """工作区是否有未提交改动（含未跟踪文件）。git 不可用/不是仓库时返回 False。"""
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT, capture_output=True, text=True, timeout=10,
        )
        return out.returncode == 0 and bool(out.stdout.strip())
    except Exception:
        return False


def code_version() -> str:
    """冻结划分 / 实验登记用的代码版本：commit 短哈希，脏工作区加 `-dirty`。"""
    commit = git_commit()
    if commit == "unknown":
        return commit
    return f"{commit}-dirty" if git_dirty() else commit
