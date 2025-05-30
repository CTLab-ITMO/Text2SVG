import argparse
from .optimization import optimize_svg_corpus


def main():
    parser = argparse.ArgumentParser(description="Optimize SVG corpus.")
    parser.add_argument('--input_dir', required=True, help="Directory containing original SVG files.")
    parser.add_argument('--corpus_dir', required=True, help="Directory to store optimized files. (ex. ./CORPUS)")
    parser.add_argument('--num_threads', type=int, default=4, help="Number of threads.")
    parser.add_argument('--timeout', type=int, default=2, help="Timeout in seconds for processing each SVG file.")
    parser.add_argument('--skip-errors', action='store_true', help="Continue processing even when some files fail.")
    parser.add_argument('--quiet', action='store_true', help="Suppress detailed progress output.")

    args = parser.parse_args()

    stage1_opts = {
        "cubic_only": True,
        "normalize_points": True,
        "normalize_scale": 256.0,
        "normalize_to_int": False,
        "timeout": args.timeout,
        "quiet": args.quiet
    }

    stage2_config_path = 'svgo.config.mjs'

    optimize_svg_corpus(
        raw_root=args.input_dir,
        corpus_root=args.corpus_dir,
        processes=args.num_threads,
        stage1_opts=stage1_opts,
        stage2_config_path=stage2_config_path,
        skip_errors=args.skip_errors,
        quiet=args.quiet
    )

if __name__ == '__main__':
    main()