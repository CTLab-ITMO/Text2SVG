import shutil
import subprocess
from functools import partial
from multiprocessing import Pool, cpu_count
from pathlib import Path
import tempfile
from typing import Iterable, List, Tuple, Optional

# ---------------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------------

def svgo_available() -> bool:
    """Return *True* if the `svgo` CLI is in PATH."""
    return shutil.which("svgo") is not None


def _ensure_svgo() -> None:
    if not svgo_available():
        raise RuntimeError("'svgo' not found. Install with: npm i -g svgo")


def _build_cmd(src: Path, dst: Path, config: Path | None) -> List[str]:
    cmd = ["svgo", str(src), "-o", str(dst)]
    if config is not None:
        cmd += ["--config", str(config)]
    return cmd


# ---------------------------------------------------------------------------
# Core single-file helper
# ---------------------------------------------------------------------------

def optimize_svg(
    src: str,
    config: Path | None = None,
) -> Optional[str]:
    _ensure_svgo()

    src = Path(src)

    with tempfile.TemporaryDirectory(prefix="svgo_tmp_") as tmpdir:
        dst = Path(tmpdir) / f"{src.stem}.svg"

        result = subprocess.run(
            _build_cmd(src, dst, config),
            capture_output=True,
            text=True,
        )
        # Если svgo упал или файл не создался — ошибка
        if result.returncode != 0 or not dst.exists():
            # print(f"[SVGO ERR] {src}: {result.stderr.strip()}")
            return None

        # Прочитаем оптимизированный файл, пока tmpdir ещё существует
        return dst.read_text(encoding="utf-8")


__all__ = [
    "svgo_available",
    "optimize_file",
    "optimize_batch",
]