# Text2SVG

This repo includes code for three key steps, required for SVG images generation via LLM.

- *src/optimization* -- SVG images optimization and cleaning;
- *src/captioning* -- generation of high-quality captions with VLM;
- *src/training* -- finetuning LLM with unsloth.

```bash
pip install .
```

## Optimization

### Prerequisites
- Install `svgo` globally:
  ```bash
  npm install -g svgo
  ```

### Data processing

```python
optimize_dir <input_dir> <output_dir> [<other args>]
```

## Captioning

## Training

Currently there are several scripts for training and evaluation. All training is handled via **Unsloth** framework.
```
