from pathlib import Path
from functools import partial
from typing import Iterable, Dict, Any, List, Union, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import json

from .stage1 import optimize_svg as optimization_stage1
from .stage2 import optimize_svg as optimization_stage2
from .hashing import hash_svg
from .util import _shard, _write_svg


# def optimize_svg_debug(
#     svg,
#     stage1_opts: Dict[str, Any] | None = None,
#     stage2_config_path: str | None = None,
# ) -> Dict[str, Any]:

#     stage1_opts = stage1_opts or {}
#     stage2_config_path = stage2_config_path or {}
    
#     stage1_result = optimization_stage1(svg, **stage1_opts)
#     if stage1_result is None:
#         raise RuntimeError("Stage-1 optimisation failed")

#     _write_svg(stage1_result, stage1_hash, corpus_root / "SVG")

#     # ------------------------------------------------------------------
#     # 2. Stage-2
#     # ------------------------------------------------------------------
    
#     stage2_result = optimization_stage2(stage1_path, config=stage2_config_path)
#     stage2_hash = hash_svg(stage2_result)


def optimize_one_svg(
    raw_path: Path,
    corpus_root: Path,
    stage1_opts: Dict[str, Any] | None = None,
    stage2_config_path: str | None = None,
) -> Dict[str, Any]:
    
    stage1_opts = stage1_opts or {}
    stage2_config_path = stage2_config_path or {}

    metadata = {
        'filename': str(raw_path),
        'raw': '',
        'stage1': '',
        'stage2': ''
    }
    
    try:
        # ------------------------------------------------------------------
        # 0. RAW
        # ------------------------------------------------------------------
        raw_svg = raw_path.read_text(encoding="utf-8")
        
        raw_hash = hash_svg(raw_svg)
        raw_path = corpus_root / "SVG" / _shard(raw_hash) / f"{raw_hash}.svg"
        
        metadata['raw'] = raw_hash
        # if raw_path.exists(): # don't saving duplicated hash
        #     print(f'Duplicate, hash: {raw_hash}.')
        #     return None
        
        _write_svg(raw_svg, raw_hash, corpus_root / "SVG")
            
        # ------------------------------------------------------------------
        # 1. Stage-1
        # ------------------------------------------------------------------

        stage1_result = optimization_stage1(raw_svg, **stage1_opts)
        if stage1_result is None:
            raise RuntimeError("Stage-1 optimisation failed")

        stage1_hash = hash_svg(stage1_result)
        stage1_path = corpus_root / "SVG" / _shard(stage1_hash) / f"{stage1_hash}.svg"
        
        metadata['stage1'] = stage1_hash
        _write_svg(stage1_result, stage1_hash, corpus_root / "SVG")

        # ------------------------------------------------------------------
        # 2. Stage-2
        # ------------------------------------------------------------------
        
        stage2_result = optimization_stage2(stage1_path, config=stage2_config_path)
        stage2_hash = hash_svg(stage2_result)
        
        metadata['stage2'] = stage2_hash
        _write_svg(stage2_result, stage2_hash, corpus_root / "SVG")
        
        return metadata
        
    except Exception as e:
        # import traceback
        # traceback.print_exc()
        # raise e
        pass
    
    return None


def _worker(
    raw_path_str: str,
    corpus_root_str: str,
    stage1_opts: Dict[str, Any],
    stage2_config_path: str,
) -> Dict[str, Any]:
    return optimize_one_svg(
        Path(raw_path_str),
        Path(corpus_root_str),
        stage1_opts=stage1_opts,
        stage2_config_path=stage2_config_path,
    )


# ─────────────────────────────────────────────────────────────────
# 1. helper: write-as-we-go
# ─────────────────────────────────────────────────────────────────
def _optimize_path_list(
    svg_paths: Sequence[Path],
    corpus_root: Path,
    stage1_opts: Dict[str, Any],
    stage2_config_path: str,
    processes: int | None,
    jsonl_file = None,          # ← new
) -> List[Dict[str, Any]]:
    svg_paths_str = [str(p) for p in svg_paths]
    results: List[Dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=processes) as ex, tqdm(
        total=len(svg_paths_str), unit="file", leave=False
    ) as bar:
        fut_to_path = {
            ex.submit(
                _worker,
                p,
                str(corpus_root),
                stage1_opts,
                stage2_config_path,
            ): p
            for p in svg_paths_str
        }

        for fut in as_completed(fut_to_path):
            res = fut.result()
            if res is not None:
                # --- write in-place -----------------------------------------
                if jsonl_file and "error" not in res:
                    jsonl_file.write(json.dumps(res, ensure_ascii=False) + "\n")
                    jsonl_file.flush()
                # ------------------------------------------------------------
                results.append(res)
            bar.update()

    return results

# ──────────────────────────────────────────────────────────────────────────────
# 4. Public API: optimise folder-by-folder & append JSONL
# ──────────────────────────────────────────────────────────────────────────────
def optimize_svg_corpus(
    raw_root: Path | str | Iterable[Path] | Iterable[str],
    corpus_root: Path | str,
    stage1_opts: Dict[str, Any] | None = None,
    stage2_config_path: str | None = None,
    processes: int | None = None,
) -> List[Dict[str, Any]]:
    stage1_opts = stage1_opts or {}
    stage2_config_path = stage2_config_path or {}

    corpus_root = Path(corpus_root)

    if isinstance(raw_root, (str, Path)):
        raw_root = Path(raw_root)
        if raw_root.is_file():
            groups: List[List[Path]] = [[raw_root]]
        else:
            subdirs = sorted([p for p in raw_root.iterdir() if p.is_dir()])
            root_svgs = list(raw_root.glob("*.svg"))
            groups = ([root_svgs] if root_svgs else []) + [
                list(d.rglob("*.svg")) for d in subdirs
            ]
    else:
        groups = [ [Path(p) for p in raw_root] ]

    # --- JSONL writer ---------------------------------------------------------
    jsonl_path: Union[Path, None] = corpus_root / "METADATA" / "hash_map.jsonl"
    jsonl_file = None
    if jsonl_path:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        jsonl_file = open(jsonl_path, "a", encoding="utf-8")

    try:
        for svg_list in groups:
            if not svg_list:
                continue

            print(f"Optimizing {len(svg_list):>5} SVGs in “{svg_list[0].parent}”")
            _optimize_path_list(
                svg_list,
                corpus_root,
                stage1_opts,
                stage2_config_path,
                processes,
                jsonl_file=jsonl_file,       # ← pass handle
            )

    finally:
        if jsonl_file:
            jsonl_file.close()
