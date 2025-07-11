from functools import partial
import hashlib
import multiprocessing as mp
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Union

# ---------------------------------------------------------------------------
# Internal: hash constructor resolver
# ---------------------------------------------------------------------------

_HashCtor = Callable[[], Any]
_DEFAULT_ALGO = "md5"
_HASHERS: dict[str, _HashCtor] = {}


def _get_hasher(algo: str) -> _HashCtor:
    """Return a *stateless* hash constructor for *algo* (lowercase)."""
    algo = algo.lower()
    if algo not in _HASHERS:
        if algo == "blake3":  # optional dependency
            try:
                import blake3  # type: ignore
            except ImportError as exc:  # pragma: no cover – optional path
                raise ValueError("blake3 requested but the package is not installed") from exc
            _HASHERS[algo] = blake3.blake3  # type: ignore[attr-defined]
        else:
            if not hasattr(hashlib, algo):  # pragma: no cover – defensive
                raise ValueError(f"Unknown hash algorithm: {algo}")
            _HASHERS[algo] = getattr(hashlib, algo)  # type: ignore[assignment]
    return _HASHERS[algo]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def hash_bytes(data: bytes, algorithm: str = _DEFAULT_ALGO) -> str:
    """Return hexadecimal digest for *data* (no newlines, always lowercase)."""
    h = _get_hasher(algorithm)()
    h.update(data)
    return h.hexdigest()


def hash_svg(svg: Union[str, bytes], algorithm: str = _DEFAULT_ALGO) -> str:
    """Hash **SVG string** – accepts *str* or UTF‑8 *bytes*."""
    if isinstance(svg, str):
        svg = svg.encode("utf-8")
    return hash_bytes(svg, algorithm)


def hash_file(path: Union[str, Path], algorithm: str = _DEFAULT_ALGO) -> str:
    """Stream‑hash a file in 8 KiB chunks (handles GB‑sized assets)."""
    path = Path(path)
    h = _get_hasher(algorithm)()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(8192), b""):  # 8 KiB
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Parallel batch hashing
# ---------------------------------------------------------------------------

def _hash_worker(svg: str, algo: str) -> str:  # multiprocessing helper
    return hash_svg(svg, algo)


def hash_iter(
    svg_iter: Iterable[str],
    *,
    algorithm: str = _DEFAULT_ALGO,
    num_proc: int | None = None,
) -> Iterator[str]:
    """Yield a hash for every SVG string in *svg_iter*.

    When ``num_proc`` > 1 the computation is parallelised via
    :pyclass:`multiprocessing.Pool`; order is preserved.
    """
    if num_proc is None or num_proc <= 1:
        for svg in svg_iter:
            yield hash_svg(svg, algorithm)
    else:
        with mp.Pool(processes=num_proc, maxtasksperchild=1) as pool:
            worker = partial(_hash_worker, algo=algorithm)
            yield from pool.imap(worker, svg_iter, chunksize=1024)


# ---------------------------------------------------------------------------
# Public re‑export list (to placate linters & __all__ users)
# ---------------------------------------------------------------------------

__all__ = [
    "hash_svg",
    "hash_file",
    "hash_iter",
]
