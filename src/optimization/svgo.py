# svgo.py
import os
import subprocess
from multiprocessing import Pool
from tqdm import tqdm
import random
import shutil
from functools import partial

def check_svgo():
    """
    Check that the 'svgo' command is available.
    """
    if not shutil.which("svgo"):
        raise SystemExit("Error: 'svgo' is not installed. Install it with: npm install -g svgo")

def process_file(task, svgo_config=None):
    """
    Run the SVGO command on a single SVG file.
    
    :param task: A tuple (input_file, output_file)
    :param svgo_config: Optional path to an SVGO configuration file. If provided,
                        the command-line argument '--config <svgo_config>' will be added.
    """
    input_file, output_file = task
    try:
        # If the output file already exists, skip processing.
        if os.path.exists(output_file):
            return
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        # Build the command.
        command = ['svgo', input_file, '-o', output_file]
        if svgo_config:
            command.extend(["--config", svgo_config])
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error processing {input_file}:\n{result.stderr}")
    except Exception as e:
        print(f"Exception processing {input_file}: {e}")

def run_svgo(input_dir, output_dir, max_samples=None, num_threads=4, svgo_config=None):
    """
    Recursively finds SVG files in `input_dir` and runs SVGO on each,
    writing the optimized files to `output_dir` (preserving directory structure).
    
    :param input_dir: Directory with input SVG files.
    :param output_dir: Directory where SVGO output will be stored.
    :param max_samples: Maximum number of files to process (randomly selected) if provided.
    :param num_threads: Number of parallel processes to run.
    :param svgo_config: Optional SVGO configuration file path.
    """
    input_dir = os.path.abspath(input_dir)
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    tasks = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith('.svg'):
                input_file = os.path.join(root, file)
                rel_path = os.path.relpath(root, input_dir)
                output_file = os.path.join(output_dir, rel_path, file)
                tasks.append((input_file, output_file))
    
    if max_samples is not None:
        tasks = random.sample(tasks, min(max_samples, len(tasks)))
    
    print(f"Running SVGO on {len(tasks)} files...")
    with Pool(num_threads) as pool:
        # Use functools.partial to create a pickleable worker function.
        worker = partial(process_file, svgo_config=svgo_config)
        for _ in tqdm(pool.imap_unordered(worker, tasks),
                      total=len(tasks),
                      desc="SVGO optimizing"):
            pass
