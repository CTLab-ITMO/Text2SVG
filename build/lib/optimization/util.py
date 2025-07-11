from pathlib import Path
from typing import Dict, List
import json

def _shard(hash_: str, prefix_len: int = 2) -> str:
    return hash_[:prefix_len]


def _write_svg(txt: str, h: str, root: Path) -> Path:
    path = root / _shard(h) / f"{h}.svg"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(txt, encoding="utf-8")
    return path


# Captions (TEXT)

def _write_caption(captions: List[str], h: str, root: Path, source: str, prompt: str) -> Path:
    if type(root) == str:
        root = Path(root)
    prefix_dir = root / _shard(h)
    prefix_dir.mkdir(parents=True, exist_ok=True)
    
    # меняем расширение на .jsonl
    path = prefix_dir / f"{h}.jsonl"
    
    # собираем уже записанные подписи, чтобы не дублировать
    existing = set()
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    existing.add(obj.get("caption"))
                except json.JSONDecodeError:
                    continue
    
    # дозаписываем новые
    with path.open("a", encoding="utf-8") as f:
        for cap in captions:
            clean = cap.strip()
            if clean and clean not in existing:
                rec = {"source": source, "caption": clean, "prompt": prompt}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                existing.add(clean)
    
    return path
    
    