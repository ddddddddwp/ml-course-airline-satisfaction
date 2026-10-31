"""跨平台清理派生数据（替代 rm -rf / find -exec，Windows 可直接跑）。

清理范围（全部可重建）：
  data/toy/              合成数据（scripts/make_toy_data.py 重建）
  data/processed/        清洗产物（make data 重建）
  data/interim/          中间产物
  reports/oof/           OOF 预测（make train 重建）
  reports/eval/          P3 评估产物
  splits/toy_fold.json   合成数据的冻结划分
  所有 __pycache__/

**绝不删除**：data/raw（唯一真实数据源）、reports/figures（图表）、reports/*.md（报告）。

用法：
    python scripts/clean.py                 # 实际清理（make clean 走这里）
    python scripts/clean.py --dry-run       # 只打印将删除的项与大小
    python scripts/clean.py --root .tmp/x   # 指定仓库根（仅测试用）

幂等：没有可清理内容时打印提示并以 0 退出；删除失败时以 1 退出。
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

try:
    from src.common.config import ROOT
except ModuleNotFoundError:  # 允许不设 PYTHONPATH 直接 python scripts/clean.py
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.common.config import ROOT

# 目录：整个删掉（相对仓库根）
TARGET_DIRS = (
    "data/toy",
    "data/processed",
    "data/interim",
    "reports/oof",
    "reports/eval",
)
# 文件：单独删掉
TARGET_FILES = ("splits/toy_fold.json",)
# 显式保护：这些位置即使有 __pycache__ 也不碰
PROTECTED = ("data/raw", "reports/figures")
# 这些目录体积大 / 与本项目缓存无关，扫描 __pycache__ 时直接跳过
SKIP_DIR_NAMES = {".git", ".venv", ".deps", ".tmp", ".pip-cache", ".idea", "node_modules"}


def _human(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    size = float(n)
    for unit in ("KB", "MB", "GB", "TB"):
        size /= 1024.0
        if size < 1024.0 or unit == "TB":
            return f"{size:.1f} {unit}"
    return f"{n} B"


def _size(p: Path) -> tuple[int, int]:
    """返回 (字节数, 文件数)；读不到的条目按 0 计。"""
    if p.is_file():
        try:
            return p.stat().st_size, 1
        except OSError:
            return 0, 0
    total = files = 0
    try:
        children = p.rglob("*")
        for child in children:
            if child.is_file():
                try:
                    total += child.stat().st_size
                    files += 1
                except OSError:
                    pass
    except OSError:
        pass
    return total, files


def _find_pycache(root: Path) -> list[Path]:
    """递归找所有 __pycache__（不跟随符号链接，跳过缓存/虚拟环境目录）。"""
    found: list[Path] = []
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            entries = list(d.iterdir())
        except OSError:
            continue
        for child in entries:
            try:
                if child.is_symlink() or not child.is_dir():
                    continue
            except OSError:
                continue
            if child.name == "__pycache__":
                found.append(child)
            elif child.name not in SKIP_DIR_NAMES:
                stack.append(child)
    return sorted(found, key=lambda p: str(p).lower())


def _collect(root: Path) -> list[Path]:
    """按固定顺序收集待删项；不存在的直接跳过（幂等）。"""
    candidates = [root / rel for rel in TARGET_DIRS]
    candidates += [root / rel for rel in TARGET_FILES]
    candidates += _find_pycache(root)

    protected = [(root / rel).resolve() for rel in PROTECTED]
    seen: set[str] = set()
    out: list[Path] = []
    for p in candidates:
        try:
            if p.is_symlink():
                continue  # 不删符号链接，避免误伤仓库外目标
            if not p.exists():
                continue
            key = str(p.resolve()).lower()
            if key in seen:
                continue
            if any(p.resolve().is_relative_to(prot) for prot in protected):
                continue
            seen.add(key)
            out.append(p)
        except OSError:
            continue
    return out


def _delete(p: Path) -> None:
    if p.is_dir() and not p.is_symlink():
        shutil.rmtree(p)
    else:
        p.unlink()


def _print_kept() -> None:
    print("[clean] 保留: data/raw（唯一真实数据源） / reports/figures（图表） / reports/*.md（报告）")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="跨平台清理派生数据（不删 data/raw、reports/figures、reports/*.md）",
    )
    ap.add_argument("--root", default=None,
                    help="仓库根目录（默认自动探测本项目根，仅测试时改）")
    ap.add_argument("--dry-run", action="store_true", help="只打印将删除的项与大小，不实际删除")
    ap.add_argument("--quiet", action="store_true", help="不逐项打印，只打印汇总")
    args = ap.parse_args(argv)

    root = Path(args.root).expanduser() if args.root else ROOT
    if not root.is_absolute():
        root = Path.cwd() / root
    root = root.resolve()

    # 安全阀：拒绝文件系统根目录，以及完全不像仓库/数据目录的路径
    if root.parent == root:
        print(f"[clean][错误] 拒绝清理文件系统根目录: {root}")
        return 2
    markers = ("Makefile", "src", "data", "reports", "splits")
    if not any((root / m).exists() for m in markers):
        print(f"[clean][错误] {root} 不像本仓库目录（缺 Makefile/src/data/reports/splits），已中止")
        return 2

    targets = _collect(root)
    mode = "DRY-RUN（只打印，不删除）" if args.dry_run else "实际删除"
    print(f"[clean] 仓库根: {root}")
    print(f"[clean] 模式  : {mode}")

    if not targets:
        print("[clean] 没有需要清理的内容（幂等）")
        _print_kept()
        return 0

    total_bytes = total_files = 0
    failures: list[tuple[str, OSError]] = []
    for p in targets:
        size, nfiles = _size(p)
        total_bytes += size
        total_files += nfiles
        rel = p.relative_to(root).as_posix()
        kind = "dir " if p.is_dir() and not p.is_symlink() else "file"
        if not args.quiet:
            print(f"  - {rel:<34} [{kind}] {nfiles:>4} 个文件  {_human(size):>9}")
        if args.dry_run:
            continue
        try:
            _delete(p)
        except OSError as e:
            failures.append((rel, e))
            print(f"[clean][警告] 删除失败 {rel}: {e}")

    verb = "将删除" if args.dry_run else "已删除"
    print(f"[clean] {verb} {len(targets)} 项 / {total_files} 个文件 / {_human(total_bytes)}")
    _print_kept()

    if failures:
        print(f"[clean][错误] {len(failures)} 项删除失败：")
        for rel, e in failures:
            print(f"  - {rel}: {e}")
        return 1
    if args.dry_run:
        print("[clean] 这是 dry-run；去掉 --dry-run 才会真正删除。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
