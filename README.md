# Text2SVG

This repo includes code for three key steps, required for SVG images generation via LLM.

- *src/optimization* -- SVG images optimization and cleaning;
- *src/captioning* -- generation of high-quality captions with VLM;
- *src/training* -- finetuning LLM with unsloth.

## Prerequisites
```bash
pip install .
npm install -g svgo
```

## Optimization

We run optimization in two stages:
- Initial optimization from raw SVG, path conversion, shifting to [[0, 256], [0, 256]] scale.
- SVGO optimization, clearing.

```bash
optimize_dir
  --input_dir        # Directory containing original SVG files.
  --output_dir       # Directory to store optimized SVGs.
  --cubic_only       # Enable conversion of segments to cubic.
  --normalize_points # Enable normalization of points.
  --normalize_scale  # Normalization scale.
  --normalize_to_int # Round coordinates to integers after normalization.
  --num_threads      # Number of threads for the Python optimization stage.
  --svgo_config      # Path to an SVGO configuration file.
```

## Captioning

We caption SVG images via VLM, Qwen/Qwen2-VL-7B-Instruct by default.

```bash
caption_dir
  --dataset     # Dataset for captioning. Should contain columns: 'svg_name' and 'svg_contents'.
  --start_index # Start index (inclusive) for the subset.
  --end_index   # End index (exclusive) for the subset.
  --batch_size  # Number of examples to process in a single batch.
  --max_samples # Limit total number of processed samples after subset selection.
  --hf_repo     # Push to this private HF dataset repo (e.g., 'username/my_repo').
  --model_path  # Path for VL model.
  --output_csv  # Path to the local CSV file where results will be stored.
```

## Training

Currently there are several scripts for training and evaluation. All training is handled via **Unsloth** framework.

```
python3 run_training.py
```


