from pathlib import Path
from functools import partial
from typing import Iterable, Dict, Any, List, Union, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import json
import tempfile

from .stage1 import optimize_svg as optimization_stage1
from .stage2 import optimize_svg as optimization_stage2
from .hashing import hash_svg
from .util import _shard, _write_svg


def optimize_svg_string(
    svg_string: str,
    stage1_opts: Dict[str, Any] | None = None,
    stage2_config_path: str | None = None,
    quiet: bool = False,
) -> str | None:
    """
    Optimize an SVG string through both stage1 and stage2 optimizations.
    
    Args:
        svg_string: The input SVG content as a string
        stage1_opts: Options for stage1 optimization
        stage2_config_path: Path to the stage2 config file
        quiet: If True, suppress output messages
    
    Returns:
        The optimized SVG string, or None if optimization fails
    """
    stage1_opts = stage1_opts or {}
    
    # Add quiet flag to stage1_opts
    if quiet and "quiet" not in stage1_opts:
        stage1_opts["quiet"] = True
    
    try:
        # Stage 1: Optimize using stage1
        if not quiet:
            print("Running stage-1 optimization...")
        
        stage1_result = optimization_stage1(svg_string, **stage1_opts)
        
        if stage1_result is None:
            if not quiet:
                print("Stage-1 optimization failed")
            return None
        
        # Stage 2: Optimize using stage2 (requires temporary file)
        if not quiet:
            print("Running stage-2 optimization...")
        
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.svg', delete=False) as tmp_file:
                tmp_file.write(stage1_result)
                tmp_file_path = tmp_file.name
            
            config_path = Path(stage2_config_path) if stage2_config_path else None
            stage2_result = optimization_stage2(tmp_file_path, config=config_path)
            
            # Clean up temporary file
            Path(tmp_file_path).unlink()
            
            if stage2_result is None:
                if not quiet:
                    print("Stage-2 optimization failed, returning stage-1 result")
                return stage1_result
            
            return stage2_result
            
        except Exception as e:
            if not quiet:
                print(f"Stage-2 optimization failed: {str(e)}, returning stage-1 result")
            return stage1_result
        
    except Exception as e:
        if not quiet:
            print(f"Optimization failed: {str(e)}")
        return None


def optimize_one_svg(
    raw_path: Path,
    corpus_root: Path,
    stage1_opts: Dict[str, Any] | None = None,
    stage2_config_path: str | None = None,
) -> Dict[str, Any]:
    
    stage1_opts = stage1_opts or {}
    quiet = stage1_opts.get("quiet", False)

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
        if not quiet:
            print(f"Processing SVG: {raw_path}")
        
        raw_hash = hash_svg(raw_svg)
        raw_path_new = corpus_root / "SVG" / _shard(raw_hash) / f"{raw_hash}.svg"
        
        metadata['raw'] = raw_hash
        _write_svg(raw_svg, raw_hash, corpus_root / "SVG")
            
        # ------------------------------------------------------------------
        # 1. Stage-1
        # ------------------------------------------------------------------
        if not quiet:
            print(f"Running stage-1 optimization for {raw_path}")
        stage1_result = optimization_stage1(raw_svg, **stage1_opts)
        
        if stage1_result is None:
            if not quiet:
                print(f"Stage-1 optimization failed for {raw_path}. Skipping this file.")
            metadata['error'] = "Stage-1 optimization failed"
            return metadata  # Return metadata with error instead of raising exception

        stage1_hash = hash_svg(stage1_result)
        stage1_path = corpus_root / "SVG" / _shard(stage1_hash) / f"{stage1_hash}.svg"
        
        metadata['stage1'] = stage1_hash
        _write_svg(stage1_result, stage1_hash, corpus_root / "SVG")

        # ------------------------------------------------------------------
        # 2. Stage-2
        # ------------------------------------------------------------------
        if not quiet:
            print(f"Running stage-2 optimization for {raw_path}")
        try:
            config_path = Path(stage2_config_path) if stage2_config_path else None
            stage2_result = optimization_stage2(str(stage1_path), config=config_path)
            
            if stage2_result is not None:
                stage2_hash = hash_svg(stage2_result)
                metadata['stage2'] = stage2_hash
                _write_svg(stage2_result, stage2_hash, corpus_root / "SVG")
        except Exception as e:
            if not quiet:
                print(f"Stage-2 optimization failed for {raw_path}: {str(e)}")
            metadata['error'] = f"Stage-2 failed: {str(e)}"
            # Continue and return metadata with what we have so far
        
        return metadata
        
    except Exception as e:
        import traceback
        if not quiet:
            print(f"\nError processing file: {raw_path}")
            print(f"SVG content preview: {raw_svg[:200]}..." if 'raw_svg' in locals() else "Could not read SVG")
            print("Full traceback:")
            traceback.print_exc()
        
        metadata['error'] = str(e)
        return metadata  # Return metadata with error instead of raising


def _worker(
    raw_path_str: str,
    corpus_root_str: str,
    stage1_opts: Dict[str, Any],
    stage2_config_path: str | None,
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
    stage2_config_path: str | None,
    processes: int | None,
    jsonl_file = None,
    skip_errors: bool = False,
    quiet: bool = False,
) -> List[Dict[str, Any]]:
    svg_paths_str = [str(p) for p in svg_paths]
    results: List[Dict[str, Any]] = []
    
    # Count of successful and failed files
    successful = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=processes) as ex, tqdm(
        total=len(svg_paths_str), 
        unit="SVG",
        desc="Optimizing SVGs",
        leave=True,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}, S:{postfix[successful]}, F:{postfix[failed]}]',
        postfix={"successful": successful, "failed": failed}
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
            try:
                res = fut.result()
                if res is not None:
                    # --- write in-place -----------------------------------------
                    if jsonl_file and "error" not in res:
                        jsonl_file.write(json.dumps(res, ensure_ascii=False) + "\n")
                        jsonl_file.flush()
                    # ------------------------------------------------------------
                    
                    # Update success/fail counters
                    if "error" in res:
                        failed += 1
                    else:
                        successful += 1
                    
                    results.append(res)
            except Exception as e:
                if skip_errors:
                    path = fut_to_path[fut]
                    if not quiet:
                        print(f"\nError processing {path}: {str(e)}")
                    failed += 1
                    results.append({'filename': path, 'error': str(e)})
                else:
                    raise
            
            # Update progress bar
            bar.postfix["successful"] = successful
            bar.postfix["failed"] = failed
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
    skip_errors: bool = False,
    quiet: bool = False,
) -> List[Dict[str, Any]]:
    stage1_opts = stage1_opts or {}

    # Add quiet flag to stage1_opts
    if quiet and "quiet" not in stage1_opts:
        stage1_opts["quiet"] = True

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

    all_results = []
    try:
        for svg_list in groups:
            if not svg_list:
                continue

            if not quiet:
                print(f"Optimizing {len(svg_list):>5} SVGs in \"{svg_list[0].parent}\"")
            
            results = _optimize_path_list(
                svg_list,
                corpus_root,
                stage1_opts,
                stage2_config_path,
                processes,
                jsonl_file=jsonl_file,
                skip_errors=skip_errors,
                quiet=quiet,
            )
            all_results.extend(results)
            
            if not quiet:
                failed = [r for r in results if 'error' in r]
                if failed:
                    print(f"\nFailed to process {len(failed)} SVGs:")
                    for i, f in enumerate(failed[:10]):  # Show first 10 failures
                        print(f"{i+1}. {f['filename']}: {f.get('error', 'Unknown error')}")
                    if len(failed) > 10:
                        print(f"... and {len(failed) - 10} more failures")

    finally:
        if jsonl_file:
            jsonl_file.close()
    
    if not quiet:
        failed = [r for r in all_results if 'error' in r]
        if failed:
            print(f"\nTotal files processed: {len(all_results)}, Failed: {len(failed)}")
    
    return all_results
