from pathlib import Path

def _shard(hash_: str, prefix_len: int = 2) -> str:
    return hash_[:prefix_len]


def _write_svg(txt: str, h: str, root: Path) -> Path:
    path = root / _shard(h) / f"{h}.svg"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(txt, encoding="utf-8")
    return path