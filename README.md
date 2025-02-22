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

```python
caption_dir
  --dataset     # Dataset for captioning. Should contain columns: 'svg_name' and 'svg_contents'.
  --start_index # Start index (inclusive) for the subset.
  --end_index   # End index (exclusive) for the subset.
  --batch_size  # Number of examples to process in a single batch.
  --max_samples # Limit total number of processed samples after subset selection.
  --hf_repo     # Push to this private HF dataset repo (e.g., 'username/my_repo').
  --model_path  # Path for VL model. [We utilize Qwen/Qwen2-VL-7B-Instruct by default].
  --output_csv  # Path to the local CSV file where results will be stored.
```

### Data processing

```python
optimize_dir <input_dir> <output_dir> [<other args>]
```

## Captioning



## Training

Currently there are several scripts for training and evaluation. All training is handled via **Unsloth** framework.
```

