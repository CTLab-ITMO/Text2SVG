# SVG Mining Tool

A tool for mining SVG files from websites using URLs from Hugging Face datasets.

## Installation

```bash
cd to/this/folder
pip install -r requirements.txt
```

## Features

- Supports any Hugging Face dataset with a URL column
- Streams data to process large datasets without loading them entirely into memory
- Filters and deduplicates SVGs
- Saves metadata (URL and timestamp) along with SVG content

## Usage

### Basic Usage

```bash
python3 main.py --dataset nhagar/fineweb_urls --output-dir /path/to/output/directory
```

### More Options

```bash
python3 main.py \
  --dataset dataset_name \
  --column url_column_name \
  --output-dir /path/to/output/directory \
  --max-concurrency 500 \ # better not set too high
  --batch-size 50000 \
  --timeout 1
```

## Parameters

- `--dataset`: Hugging Face dataset name (required)
- `--output-dir`: Directory where SVG files will be saved (required)
- `--column`: Column name containing URLs (default: "url")
- `--max-concurrency`: Maximum concurrent HTTP requests (default: 500)
- `--batch-size`: Batch size for URL processing (default: 50000)
- `--timeout`: Timeout in seconds for HTTP requests (default: 1)
- `--start-offset`: Skip this many URLs before processing (default: 0)
- `--debug`: Enable debug mode with detailed logging

## Output

SVGs are saved to the specified output directory in JSONL format with the following structure:

```json
{
  "url": "https://example.com/page",
  "timestamp": "2024-03-21T10:30:45.123456",
  "svg": "<svg>...</svg>"
}
```

Files are organized in subfolders by the first two characters of SVGs MD5 hash.

## Example

Example of mined svgs is available at [HF](https://huggingface.co/datasets/VectorGraphics/open-svg).
