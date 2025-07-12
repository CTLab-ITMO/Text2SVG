# Text2SVG

🌐 Available languages: [English](README.md) | [Русский](README.ru.md)

A comprehensive toolkit for SVG mining, optimization, captioning, and text-to-SVG generation. 
The project consists of three main components working together to create a complete pipeline from raw SVG collection to AI-powered SVG generation.

```
Text2SVG
├── svg-corpus         # Mining, optimization, and captioning
│   ├── mining         # Web scraping SVG files
│   ├── optimization   # Two-stage SVG optimization
│   └── captioning     # VLM-based SVG description generation
├── llm4svg            # Training and evaluation scripts
└── svg-encoder        # Model architecture and inference
```

## Quick Start

```bash
git clone https://github.com/CTLab-ITMO/Text2SVG
cd Text2SVG
pip install .
apt update && apt install libcairo2
```

## SVG Corpus Pipeline

### Mining

A tool for mining SVG files from websites using URLs from HuggingFace datasets. Collects approximately 4 million URLs per month, considering deduplication.

<img src="https://github.com/CTLab-ITMO/Text2SVG/blob/main/imgs/One-Month%20Extrapolation.png" alt="Growth projection showing sublinear scaling due to deduplication" width="600"/>

#### Installation

Ensure, you have run quick start installation steps.

```bash
cd Text2SVG/svg-corpus/mining
pip install -r requirements.txt
```

#### Basic Usage

You can choose any dataset from the HuggingFace collection that includes a column named url containing URLs.

```bash
python3 main.py --dataset nhagar/fineweb_urls --output-dir /path/to/output
```

#### Advanced Options

Depending on your system you can change arguments:

- **`--max-concurrency`** (default: 500): Controls the maximum number of simultaneous HTTP requests. This is the key parameter for balancing speed vs. system resources.

- **`--batch-size`** (default: 50000): Number of URLs processed together before yielding control.

- **`--timeout`** (default: 1): Maximum time in seconds to wait for each HTTP request. Aggressive timeouts improve throughput by skipping slow sites.


```bash
python3 main.py \
  --dataset nhagar/fineweb_urls \
  --column url \
  --output-dir /path/to/output \
  --max-concurrency 500 \
  --batch-size 50000 \
  --timeout 1 \
  --start-offset 0 \
  --debug
```

#### Output Format

SVGs are saved in JSONL format, organized by MD5 hash prefixes:

```json
{
  "url": "https://example.com/page",
  "timestamp": "2024-03-21T10:30:45.123456",
  "svg": "<svg>...</svg>"
}
```

**Example Dataset:** 

Small part of collected SVGs is available via [link](https://huggingface.co/datasets/VectorGraphics/open-svg).

### Optimization

Two-stage optimization pipeline for cleaning and normalizing SVG files:

1. **Stage 1:** Path conversion, coordinate normalization to [0, 256] scale
2. **Stage 2:** SVGO optimization for size reduction and cleanup

#### Python API

```python
from optimization import optimize_svg_string

# Basic optimization
result = optimize_svg_string(svg_string)

# With custom options
result = optimize_svg_string(
    svg_string=svg_text,
    stage1_opts={
        "normalize_scale": 256.0,
        "cubic_only": True,
        "normalize_to_int": False
    },
    stage2_config_path='./svgo.config.mjs',
    quiet=True
)
```

#### Command Line Interface

```bash
optimize_dir \
  --input_dir ./raw_svgs \
  --output_dir ./optimized_svgs \
  --cubic_only \
  --normalize_points \
  --normalize_scale 256 \
  --num_threads 4 \
  --svgo_config ./svgo.config.mjs
```

#### Optimization Example

**Before optimization (648 characters):**
```xml
<svg version="1.1" xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 20 20">
<title>power</title>
<path d="M10.625 1.681c0-0.345-0.28-0.625-0.625-0.625s-0.625 0.28-0.625 0.625v8.125c0 0.345 0.28 0.625 0.625 0.625s0.625-0.28 0.625-0.625v-8.125z"></path>
<path d="M7.12 2.881c0.318-0.135 0.466-0.502 0.33-0.82s-0.502-0.466-0.82-0.33c-3.156 1.343-5.38 4.436-5.38 8.075 0 4.845 3.905 8.75 8.75 8.75s8.75-3.905 8.75-8.75c0-3.639-2.225-6.732-5.38-8.075-0.318-0.135-0.685 0.013-0.82 0.33s0.013 0.685 0.33 0.82c2.719 1.157 4.62 3.814 4.62 6.925 0 4.155-3.345 7.5-7.5 7.5s-7.5-3.345-7.5-7.5c0-3.111 1.9-5.768 4.62-6.925z"></path>
</svg>
```

<img src="https://github.com/CTLab-ITMO/Text2SVG/blob/main/imgs/example_1_before_optimization.png?raw=true" alt="Before optimization" width="200"/>

**After optimization (343 characters, 47% reduction):**
```xml
<svg viewBox="0 0 256 256">
  <path d="M136 22Q135 14 128 14T120 22V126Q121 133 128 134 135 133 136 126z"/>
  <path d="M91 37Q98 33 95 26 92 20 85 22C44 39 16 79 16 126 16 188 66 238 128 238S240 188 240 126C240 79 212 39 171 22Q164 20 161 26 158 33 165 37C200 52 224 86 224 126 224 179 181 222 128 222S32 179 32 126C32 86 56 52 91 37"/>
</svg>
```

<img src="https://github.com/CTLab-ITMO/Text2SVG/blob/main/imgs/example_1_after_optimization.png?raw=true" alt="After optimization" width="200"/>

### Captioning

Vision-language model (VLM) based SVG description generation using [LMDeploy](https://github.com/InternLM/lmdeploy). The pipeline supports any vision-language model available through LMDeploy, including InternVL series, Qwen2-VL, and other compatible models.

#### Installation

Ensure you have run the quick start installation steps, then install the captioning dependencies:

```bash
cd Text2SVG/svg-corpus/captioning
pip install lmdeploy timm
```

#### Basic Usage

```bash
python caption_intern.py \
  --dataset VectorGraphics/svg-corpus \
  --output captions.jsonl \
  --model OpenGVLab/InternVL3-2B
```

#### Advanced Usage

```bash
python caption_intern.py \
  --dataset VectorGraphics/svg-corpus \
  --output detailed_captions.jsonl \
  --model OpenGVLab/InternVL3-8B \
  --batch-size 32 \
  --resolution 224 \
  --temperature 0.1 \
  --tp 1 \
  --max-samples 100_000
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model` | str | `OpenGVLab/InternVL3-2B` | Model name or path (any LMDeploy-compatible VLM) |
| `--dataset` | str | `VectorGraphics/svg-corpus` | Dataset name or local JSON file path |
| `--data-files` | str | `*_stage2*` | Data files pattern for HuggingFace datasets |
| `--output` | str | `augmented_captioned_internvl3_2b.jsonl` | Output JSONL file path |
| `--resolution` | int | `224` | Image resolution for processing |
| `--batch-size` | int | `64` | Processing batch size |
| `--temperature` | float | `0.01` | Generation temperature (0.0-1.0) |
| `--tp` | int | `1` | Tensor parallel size for multi-GPU |
| `--session-len` | int | `16384` | Model session length |
| `--timeout` | int | `1` | SVG to PNG conversion timeout |
| `--start-index` | int | `0` | Start processing from this index |
| `--max-samples` | int | `None` | Limit total samples processed |
| `--seed` | int | `12343` | Random seed for reproducibility |

#### Output Format

Captions are saved in JSONL format:

```json
{
  "hash": "a1b2c3d4...",
  "caption": "A blue circular icon with a power symbol",
  "source": "InternVL3-2B",
  "prompt": "Describe this object in 10 words or less, concise style. Response in English."
}
```

## LLM Fine-tuning

### Current Generation Examples

<img src="https://github.com/CTLab-ITMO/Text2SVG/blob/main/imgs/generated_comp.png?raw=true" alt="Generated SVG examples" width="600"/>

### Training

Fine-tuning is powered by the [Unsloth](https://github.com/unslothai/unsloth) framework.

#### Prerequisites

```bash
cd Text2SVG/llm4svg
pip install unsloth transformers datasets wandb
```

#### Training Configuration

The training script (`run_training.py`) uses the following key parameters:

- **Base Model**: Qwen2.5-Coder-32B-Instruct (configurable)
- **Dataset**: VectorGraphics/svg-corpus
- **LoRA Configuration**: r=128, alpha=32, targeting all attention layers
- **Training**: 1 epoch, batch size 20, gradient accumulation 8 steps
- **Optimization**: AdamW 8-bit with cosine scheduler
- **Memory**: 4-bit quantization with gradient checkpointing

#### Running Training

```bash
python3 run_training.py
```

#### Customization

Modify these parameters in `run_training.py`:

```python
# Model selection
model_name = "Qwen2.5-Coder-32B-Instruct"  # or any supported model

# LoRA configuration
r = 128                    # LoRA rank
lora_alpha = 32           # LoRA alpha
lora_dropout = 0          # LoRA dropout

# Training parameters
per_device_train_batch_size = 20
gradient_accumulation_steps = 8
learning_rate = 5e-5
num_train_epochs = 1
```

### Inference

Generate SVG images from text descriptions using the fine-tuned model:

```bash
python3 run_eval.py
```

#### Example Usage

Edit the prompt in `run_eval.py`:

```python
inputs = tokenizer([
    inf_prompt.format(
        "A blue circular button with power symbol",  # Your caption
        "",  # Leave empty for generation
    )
], return_tensors="pt").to("cuda")
```

## SVG Encoder

A module for encoding SVGs (obtaining embeddings of vector images) is [here](svg-encoder).

The training is based on [ModernBERT](https://github.com/AnswerDotAI/ModernBERT).

### Training Process

The SVG encoder model was trained in original ModernBERT settings on a large SVG corpus:
- First stage: 12 billion tokens
- Second stage: 3 billion tokens 
- Third stage: 2 billion tokens

### Usage (current scripts)

Convert trained models to Hugging Face format:

```bash
python scripts/convert_to_hf.py \
  --config path/to/config \
  --checkpoint path/to/checkpoint \
  --tokenizer path/to/tokenizer \
  --output path/to/save \
  --save-tokenizer
```

### Acknowledgements

- [ModernBERT](https://github.com/AnswerDotAI/ModernBERT) for providing the base architecture

## License

## Citation
