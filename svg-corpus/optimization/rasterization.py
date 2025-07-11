import io
import os
import signal
from contextlib import contextmanager
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Iterable, List, Tuple, Union

import re
from cairosvg import svg2png
from PIL import Image, ImageStat
from .util import _shard

# ---------------------------------------------------------------------------
# Timeout handling (Unix‑only; on Windows calls run without timeout)
# ---------------------------------------------------------------------------

class TimeoutException(Exception):
    """Raised when CairoSVG takes longer than *timeout* seconds."""


@contextmanager
def _alarm(seconds: int):
    if os.name != "posix" or seconds <= 0:
        yield
        return
    def _timeout_handler(signum, frame):
        raise TimeoutException()

    old_handler = signal.signal(signal.SIGALRM, _timeout_handler) 
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


# ---------------------------------------------------------------------------
# Sanitize helpers
# ---------------------------------------------------------------------------

def sanitize(svg: str) -> str:
    """Fix common SVG issues so that CairoSVG can render them reliably."""
    svg = svg.replace("xlink:href", "href")
    svg = re.sub(
        r"<(\/?)(lineargradient|radialgradient|clippath|mask|filter)",
        lambda m: f"<{m.group(1)}{m.group(2).title()}",
        svg,
        flags=re.I,
    )
    # add viewBox if missing
    if "viewBox" not in svg and "viewbox" not in svg:
        w = re.search(r"width=\"([\d\.]+)", svg)
        h = re.search(r"height=\"([\d\.]+)", svg)
        if w and h:
            svg = svg.replace(
                "<svg", f"<svg viewBox=\"0 0 {w.group(1)} {h.group(1)}\"", 1
            )
    return svg


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def is_blank(img: Image.Image, thr: int = 5) -> bool:
    """Return *True* if **img** is nearly blank (std‑dev < `thr`) or fully transparent."""
    if img.mode in ("RGBA", "LA") and max(img.split()[-1].getextrema()) < thr:
        return True
    return ImageStat.Stat(img.convert("L")).stddev[0] < thr


# ---------------------------------------------------------------------------
# Core rasterisation
# ---------------------------------------------------------------------------

def svg_to_png_bytes(
    svg: Union[str, bytes],
    *,
    size: int = 512,
    timeout: int = 30,
) -> bytes | None:
    """Convert **svg** to PNG and return bytes; *None* if render fails or blank."""
    raw_svg = sanitize(svg.decode("utf-8") if isinstance(svg, bytes) else svg)
    try:
        with _alarm(timeout):
            png_bytes = svg2png(
                bytestring=raw_svg.encode("utf-8"),
                output_width=size,
                output_height=size,
            )
    except TimeoutException:
        return None
    except Exception:
        # CairoSVG can throw many different exceptions on malformed input.
        return None

    # post‑process via Pillow (make opaque white background, blank check)
    try:
        img = Image.open(io.BytesIO(png_bytes))
    except Exception:  # corrupt stream
        return None

    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg
    else:
        img = img.convert("RGB")

    if is_blank(img):
        return None

    out = io.BytesIO()
    img.save(out, format="PNG", optimize=True)
    return out.getvalue()


# ---------------------------------------------------------------------------
# Disk helpers
# ---------------------------------------------------------------------------

def save_png(hash_: str, png_bytes: bytes, png_root: Path, *, prefix_len: int = 2) -> Path:
    """Write *png_bytes* to *png_root/<aa>/<hash>.png* and return the path."""
    out_dir = png_root / _shard(hash_, prefix_len)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{hash_}.png"
    path.write_bytes(png_bytes)
    return path


# ---------------------------------------------------------------------------
# Record‑based helpers (hash, svg)
# ---------------------------------------------------------------------------

def rasterize_record(
    record: Tuple[str, str],
    png_root: Path,
    *,
    size: int = 512,
    timeout: int = 30,
) -> Path | None:
    """Render a single *(hash, svg)* pair to PNG on disk.

    Returns the output path on success or *None* if skipped/failed.
    """
    h, svg = record
    out_path = png_root / _shard(h) / f"{h}.png"
    if out_path.exists():
        return out_path  # already rendered

    png_bytes = svg_to_png_bytes(svg, size=size, timeout=timeout)
    if png_bytes is None:
        return None

    return save_png(h, png_bytes, png_root)


def _worker(record: Tuple[str, str], png_root: str, size: int, timeout: int) -> str | None:
    # helper for Pool imap
    path = rasterize_record(record, Path(png_root), size=size, timeout=timeout)
    return str(path) if path else None


def rasterize_batch(
    records: Iterable[Tuple[str, str]],
    png_root: Path,
    *,
    size: int = 512,
    timeout: int = 30,
    num_proc: int | None = None,
) -> List[Path]:
    """Rasterise many *(hash, svg)* pairs in parallel and return paths of written PNGs."""
    png_root = Path(png_root)
    if num_proc is None or num_proc <= 1:
        written: List[Path] = []
        for rec in records:
            path = rasterize_record(rec, png_root, size=size, timeout=timeout)
            if path:
                written.append(path)
        return written

    # Parallel branch
    _args = (str(png_root), size, timeout)
    with Pool(processes=num_proc or cpu_count(), maxtasksperchild=1) as pool:
        results = pool.starmap(_worker, ((rec, *_args) for rec in records), chunksize=128)
    return [Path(p) for p in results if p]


# ---------------------------------------------------------------------------
# Public re‑export list
# ---------------------------------------------------------------------------

__all__ = [
    "sanitize",
    "is_blank",
    "svg_to_png_bytes",
    "save_png",
    "rasterize_record",
    "rasterize_batch",
]
