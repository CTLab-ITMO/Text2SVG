#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, io, json, random, re, gc, argparse
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("MKL_THREADING_LAYER", "GNU")

import datasets
from PIL import Image
from tqdm import tqdm
from optimization.rasterization import svg_to_png_bytes

from lmdeploy import (pipeline, TurbomindEngineConfig,
                      ChatTemplateConfig, GenerationConfig)
from lmdeploy.vl.constants import IMAGE_TOKEN

STYLES = ["concise", "detailed", "technical", "schematic"]
FOCUSES = ["object", "scene", "color", "action"]
MAX_WORDS = [5, 10, 15, 20, 50]

def ask(style, focus, max_words):
    return f"Describe this {focus} in {max_words} words or less, {style} style. Response in English."

def main():
    parser = argparse.ArgumentParser(description="Generate SVG captions using vision-language models")
    
    parser.add_argument("--model", default="OpenGVLab/InternVL3-2B",
                        help="Model name or path (default: OpenGVLab/InternVL3-2B)")
    parser.add_argument("--dataset", default="VectorGraphics/svg-corpus",
                        help="Input dataset name or path (default: VectorGraphics/svg-corpus)")
    parser.add_argument("--data-files", default="*_stage2*",
                        help="Data files pattern to load (default: *_stage2*)")
    parser.add_argument("--output", default="augmented_captioned_internvl3_2b.jsonl",
                        help="Output JSONL file path (default: augmented_captioned_internvl3_2b.jsonl)")
    parser.add_argument("--resolution", type=int, default=224,
                        help="Image resolution for processing (default: 224)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for processing (default: 64)")
    parser.add_argument("--source", default="InternVL3-2B",
                        help="Source identifier for output (default: InternVL3-2B)")
    parser.add_argument("--temperature", type=float, default=0.01,
                        help="Generation temperature (default: 0.01)")
    parser.add_argument("--tp", type=int, default=1,
                        help="Tensor parallel size for multi-GPU (default: 1)")
    parser.add_argument("--session-len", type=int, default=16384,
                        help="Session length for the model (default: 16384)")
    parser.add_argument("--timeout", type=int, default=1,
                        help="Timeout for SVG to PNG conversion (default: 1)")
    parser.add_argument("--seed", type=int, default=12343,
                        help="Random seed for reproducibility (default: 12343)")
    parser.add_argument("--start-index", type=int, default=0,
                        help="Start index for dataset processing (default: 0)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to process (default: all)")
    
    args = parser.parse_args()
    
    print(f"⏳ Loading {args.model} into TurboMind …")
    pipe = pipeline(
        args.model,
        backend_config=TurbomindEngineConfig(
            tp=args.tp,
            session_len=args.session_len
        ),
        chat_template_config=ChatTemplateConfig(model_name="internvl2_5")
    )
    gen_cfg = GenerationConfig(temperature=args.temperature)
    
    # Load dataset
    if Path(args.dataset).exists():
        # Local dataset path
        ds = datasets.load_dataset("json", data_files=str(args.dataset), split="train")
    else:
        # HuggingFace dataset
        ds = datasets.load_dataset(
            args.dataset,
            split="train",
            data_files=[args.data_files] if args.data_files != "*_stage2*" else ["*_stage2*"]
        )
    
    ds = ds.shuffle()
    
    # Apply start index and max samples
    if args.start_index > 0:
        ds = ds.select(range(args.start_index, len(ds)))
    
    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    
    print(f"📚 Dataset size: {len(ds)} samples")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    
    total_processed = 0
    total_saved = 0
    
    for start in tqdm(range(0, len(ds), args.batch_size), desc="Batches"):
        sub = [ds[i] for i in range(start, min(start + args.batch_size, len(ds)))]
        prompts, metas = [], []
        
        for item in sub:
            svg_bytes = (item["svg"].encode() if isinstance(item["svg"], str)
                         else item["svg"])
            try:
                png = svg_to_png_bytes(svg_bytes, size=args.resolution, timeout=args.timeout)
            except Exception:
                continue
            if not png:
                continue
            
            img = Image.open(io.BytesIO(png)).convert("RGB")
            q = ask(random.choice(STYLES),
                    random.choice(FOCUSES),
                    random.choice(MAX_WORDS))
            
            prompts.append((q, img))
            metas.append({"hash": item["hash"], "question": q})
        
        if not prompts:
            continue
        
        responses = pipe(prompts, gen_config=gen_cfg)
        total_processed += len(responses)
        
        kept = 0
        with output_path.open("a", encoding="utf-8") as fout:
            for meta, resp in zip(metas, responses):
                caption = resp.text.strip()
                caption = re.sub(r"^(Caption:|Description:|Image shows:)\s*", "",
                                 caption, flags=re.I)
                if caption:
                    fout.write(json.dumps({
                        "hash": meta["hash"],
                        "caption": caption,
                        "source": args.source,
                        "prompt": meta["question"]
                    }, ensure_ascii=False) + "\n")
                    kept += 1
        
        total_saved += kept
        tqdm.write(f"Batch {start//args.batch_size}: saved {kept}/{len(responses)}")
    
    print(f"\n✅ Finished! Processed {total_processed} samples, saved {total_saved} captions")
    print(f"📄 Output saved to: {output_path}")
    
    pipe.close()
    gc.collect()

if __name__ == "__main__":
    main()
