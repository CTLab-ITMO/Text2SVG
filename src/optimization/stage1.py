import os
import signal
import traceback
from contextlib import contextmanager
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Iterable, List, Tuple
import xml.etree.ElementTree as etree

from .hashing import hash_svg
from .rasterization import _shard
from .svg_core import postfix_svg_root, optimize_svg_from_str

# ---------------------------------------------------------------------------
# Timeout guard (reuse rasterisation logic but shorter default)
# ---------------------------------------------------------------------------

class TimeoutException(Exception):
    pass


@contextmanager
def _alarm(seconds: int):
    if os.name != "posix" or seconds <= 0:
        yield
        return

    def _handler(signum, frame):  # noqa: ANN001 – signal handler signature
        raise TimeoutException()

    old = signal.signal(signal.SIGALRM, _handler)  # type: ignore[arg-type]
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Core single‑SVG optimiser
# ---------------------------------------------------------------------------

def _postprocess_root(root: etree.Element) -> str:
    """XML serialisation + fix *viewBox* casing + re‑attach xmlns."""
    postfix_svg_root(root)
    svg_txt = etree.tostring(root, encoding="utf-8").decode("utf-8")
    return svg_txt.replace("viewbox", "viewBox")


_DEFAULT_OPTS = dict(
    cubic_only=True,
    normalize_points=True,
    normalize_scale=256.0,
    normalize_to_int=False,
)


def optimize_svg(svg: str, timeout: int = 2, **opts) -> str | None:
    """Optimise **svg** string and return the cleaned SVG string.

    If optimisation fails (fatal parse error, timeout, or resulting SVG empty)
    the function returns *None* so callers can skip the sample.
    """
    cfg = _DEFAULT_OPTS.copy()
    cfg.update(opts)

    try:
        with _alarm(timeout):
            root = optimize_svg_from_str(svg, **cfg)
    except (TimeoutException, Exception):  # broad on purpose, we log below
        # traceback.print_exc(limit=1)
        return None

    try:
        return _postprocess_root(root)
    except Exception:  # XML serialisation failed
        # traceback.print_exc(limit=1)
        return None


# ---------------------------------------------------------------------------
# Disk helpers
# ---------------------------------------------------------------------------

def _write_svg(hash_: str, svg_txt: str, out_root: Path, prefix_len: int = 2) -> Path:
    out_dir = out_root / _shard(hash_, prefix_len)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{hash_}.svg"
    path.write_text(svg_txt, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# What we export
# ---------------------------------------------------------------------------

__all__ = [
    "optimize_svg",
    "hash_and_optimize",
]
