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

### Example

Before optimization:

<img src="https://github.com/CTLab-ITMO/Text2SVG/blob/main/imgs/example_1_before_optimization.png?raw=true" alt="Initial image" width="200"/>


```
<svg version="1.1" xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20">
<title>power</title>
<path d="M10.625 1.681c0-0.345-0.28-0.625-0.625-0.625s-0.625 0.28-0.625 0.625v8.125c0 0.345 0.28 0.625 0.625 0.625s0.625-0.28 0.625-0.625v-8.125z"></path>
<path d="M7.12 2.881c0.318-0.135 0.466-0.502 0.33-0.82s-0.502-0.466-0.82-0.33c-3.156 1.343-5.38 4.436-5.38 8.075 0 4.845 3.905 8.75 8.75 8.75s8.75-3.905 8.75-8.75c0-3.639-2.225-6.732-5.38-8.075-0.318-0.135-0.685 0.013-0.82 0.33s0.013 0.685 0.33 0.82c2.719 1.157 4.62 3.814 4.62 6.925 0 4.155-3.345 7.5-7.5 7.5s-7.5-3.345-7.5-7.5c0-3.111 1.9-5.768 4.62-6.925z"></path>
```

After optimization:

<img src="https://github.com/CTLab-ITMO/Text2SVG/blob/main/imgs/example_1_after_optimization.png?raw=true" alt="Optimized image" width="200"/>

```
<svg viewBox="0 0 256 256">
  <path d="M136 22Q135 14 128 14T120 22V126Q121 133 128 134 135 133 136 126z"/>
  <path d="M91 37Q98 33 95 26 92 20 85 22C44 39 16 79 16 126 16 188 66 238 128 238S240 188 240 126C240 79 212 39 171 22Q164 20 161 26 158 33 165 37C200 52 224 86 224 126 224 179 181 222 128 222S32 179 32 126C32 86 56 52 91 37"/>
</svg>
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

### Example

*Generated caption:* Black power button with a diagonal line. The button has a circular shape with a rectangular line bisecting it.

<img src="https://github.com/CTLab-ITMO/Text2SVG/blob/main/imgs/example_1_after_optimization.png?raw=true" alt="Optimized image" width="200"/>


## Training

Currently there are several scripts for training and evaluation. All training is handled via **Unsloth** framework.

```
python3 run_training.py
```

### Example of current generation

<img src="https://github.com/CTLab-ITMO/Text2SVG/blob/main/imgs/generated_comp.png?raw=true" alt="Initial image"/>

