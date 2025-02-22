#!/usr/bin/env python3
import os
import sys
import xml.etree.ElementTree as etree
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

from .svg_optimizer import optimize_svg_from_file, postfix_svg_root
from .svgo import check_svgo, run_svgo

def optimize_and_save_svg(
    input_svg_path: str,
    output_svg_path: str,
    cubic_only: bool = True,
    normalize_points: bool = True,
    normalize_scale: float = 256,
    normalize_to_int: bool = False
):
    """
    Reads an SVG from 'input_svg_path', optimizes it using Python logic, and writes the result
    to 'output_svg_path'. Ensures that necessary subdirectories are created.
    """
    # Optimize the SVG using your custom routine.
    root = optimize_svg_from_file(
        filename=input_svg_path,
        cubic_only=cubic_only,
        normalize_points=normalize_points,
        normalize_scale=normalize_scale,
        normalize_to_int=normalize_to_int
    )
    # Adjust the SVG root.
    postfix_svg_root(root)
    
    # Serialize the optimized SVG and fix the case for "viewBox".
    svg_optimized = etree.tostring(root, encoding='utf-8').decode("utf-8")
    svg_optimized = svg_optimized.replace('viewbox', 'viewBox')
    
    # Ensure output directory exists and write the file.
    os.makedirs(os.path.dirname(output_svg_path), exist_ok=True)
    with open(output_svg_path, "w", encoding="utf-8") as f:
        f.write(svg_optimized)

def optimize_dir(
    input_dir: str,
    output_dir: str,
    cubic_only: bool = True,
    normalize_points: bool = True,
    normalize_scale: float = 256,
    normalize_to_int: bool = False,
    num_threads: int = 32
):
    """
    Recursively finds all .svg files in 'input_dir', applies the Python-based optimization,
    and writes the optimized files to 'output_dir' preserving the directory structure.
    """
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Collect all .svg files.
    svg_files = []
    for root_path, _, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith(".svg"):
                svg_files.append(os.path.join(root_path, filename))
    
    print(f"Found {len(svg_files)} SVG files to optimize.")
    
    lock = threading.Lock()
    
    def process_svg(svg_path):
        relative_path = os.path.relpath(svg_path, start=input_dir)
        output_path = os.path.join(output_dir, relative_path)
        try:
            optimize_and_save_svg(
                input_svg_path=svg_path,
                output_svg_path=output_path,
                cubic_only=cubic_only,
                normalize_points=normalize_points,
                normalize_scale=normalize_scale,
                normalize_to_int=normalize_to_int
            )
        except Exception as e:
            with lock:
                print(f"Error processing '{svg_path}': {e}")
        finally:
            with lock:
                pbar.update(1)
    
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        with tqdm(total=len(svg_files), desc="Optimizing SVGs") as pbar:
            futures = [executor.submit(process_svg, svg_path) for svg_path in svg_files]
            for future in futures:
                future.result()

def main():
    parser = argparse.ArgumentParser(
        description="Optimize SVGs with Python and then further optimize them with SVGO."
    )
    parser.add_argument("input_dir", help="Directory containing original SVG files.")
    parser.add_argument("output_dir", help="Directory to store optimized SVGs.")
    # parser.add_argument("final_output_dir", help="Directory to store final SVGO-optimized SVGs.")
    
    parser.add_argument("--cubic_only", dest="cubic_only", action="store_true",
                        help="Enable conversion of segments to cubic (default: enabled).", default=True)
    
    parser.add_argument("--normalize_points", dest="normalize_points", action="store_true",
                        help="Enable normalization of points.", default=True)
    
    parser.add_argument("--normalize_scale", type=float, default=256.0,
                        help="Normalization scale.")
    parser.add_argument("--normalize_to_int", action="store_true",
                        help="Round coordinates to integers after normalization.", default=False)
    parser.add_argument("--num_threads", type=int, default=4,
                        help="Number of threads for the Python optimization stage.")
    
    parser.add_argument("--svgo_config", type=str, default=None,
                        help="Path to an SVGO configuration file (e.g., svgo.config.mjs). "
                             "If provided, SVGO will use it via the '--config' option.")
    
    # parser.add_argument("--max_samples", type=int, default=None,
    #                     help="Maximum number of files to process with SVGO.")
    
    args = parser.parse_args()
    
    # --- Step 1: Python-based optimization ---
    print("\n[Step 1] Starting Python-based SVG optimization...")
    optimize_dir(
        input_dir=args.input_dir,
        output_dir=f"{args.output_dir}/normalized",
        cubic_only=args.cubic_only,
        normalize_points=args.normalize_points,
        normalize_scale=args.normalize_scale,
        normalize_to_int=args.normalize_to_int,
        num_threads=args.num_threads
    )
    print("[Step 1] Python-based optimization completed.\n")
    
    # --- Step 2: SVGO optimization ---
    print("[Step 2] Starting SVGO optimization...")
    check_svgo()
    run_svgo(
        input_dir=f"{args.output_dir}/normalized",
        output_dir=f"{args.output_dir}/svgo_optimized",
        # max_samples=args.max_samples,
        num_threads=args.num_threads,
        svgo_config=args.svgo_config
    )
    print("[Step 2] SVGO optimization completed.")
    print("\nAll steps completed successfully.")

if __name__ == "__main__":
    main()
