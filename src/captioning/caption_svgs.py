#!/usr/bin/env python3

import argparse
import logging
import io
import csv
import torch
from PIL import Image
import cairosvg
from tqdm import tqdm
from huggingface_hub import HfApi
from datasets import load_dataset
from huggingface_hub import login, create_repo
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor

logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Select a range from a large HF dataset, batch-process with Qwen2-VL-7B-Instruct, then optionally push to a private repo."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Dataset for captioning. Should contain columns: 'svg_name' and 'svg_contents'."
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Start index (inclusive) for the subset."
    )
    parser.add_argument(
        "--end_index",
        type=int,
        default=-1,
        help="End index (exclusive) for the subset."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of examples to process in a single batch."
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="If specified, limit total number of processed samples after subset selection."
    )
    parser.add_argument(
        "--hf_repo",
        type=str,
        default=None,
        help="If specified, push to this private HF dataset repo (e.g., 'username/my_repo')."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen2-VL-7B-Instruct",
        help="Path for VL model."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="captions.csv",
        help="Path to the local CSV file where results will be stored."
    )
    args = parser.parse_args()
    return args

def load_image_from_svg_content(svg_content, target_size=(256, 256)):
    """
    Convert SVG to a PIL image with a white background.
    """
    try:
        png_data = cairosvg.svg2png(
            bytestring=svg_content.encode("utf-8"),
            output_width=target_size[0],
            output_height=target_size[1],
        )
        image = Image.open(io.BytesIO(png_data))
        if image.mode == "RGBA":
            bg = Image.new("RGB", image.size, (255, 255, 255))
            bg.paste(image, mask=image.split()[3])
            image = bg
        else:
            image = image.convert("RGB")
        return image
    except Exception as e:
        logging.error(f"SVG to image conversion failed: {e}")
        return None

def main():
    args = parse_args()

    dataset = args.dataset
    start_idx = args.start_index
    end_idx = args.end_index
    batch_size = args.batch_size
    max_samples = args.max_samples
    model_path = args.model_path

    # ----------------------------------------------------------------------
    # 2) Load the entire dataset (non-streaming), then select a subset
    # ----------------------------------------------------------------------

    logging.info(f"Loading the dataset {dataset}.")
    # hf_dataset = load_dataset("TatonkaHF/SVGs_vz_processed_2.8M", split="train")
    hf_dataset = load_dataset(dataset, split="train")
    total_len = len(hf_dataset)
    logging.info(f"Original dataset size: {total_len}")

    if end_idx == -1:
        end_idx = total_len

    # Clip end_idx if it exceeds the dataset length
    end_idx = min(end_idx, total_len)

    # Subset the dataset
    if start_idx >= end_idx:
        raise ValueError(f"start_index ({start_idx}) must be less than end_index ({end_idx}).")
    
    logging.info(f"Selecting dataset range [{start_idx}, {end_idx})...")
    hf_dataset = hf_dataset.select(range(start_idx, end_idx))
    logging.info(f"Subset size: {len(hf_dataset)}")

    # If max_samples is specified, reduce further
    if max_samples is not None and max_samples < len(hf_dataset):
        logging.info(f"Further limiting to max_samples={max_samples}.")
        hf_dataset = hf_dataset.select(range(max_samples))

    logging.info(f"Final subset size: {len(hf_dataset)}")

    # ----------------------------------------------------------------------
    # 3) Load Qwen2-VL-7B-Instruct model & processor
    # ----------------------------------------------------------------------
    logging.info(f"Loading model from {model_path}...")
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)

    # Define the prompt text
    prompt_text = r'''
**Instruction:**
Generate a concise caption for the provided image that highlights the key objects within it. 
The caption should consist of:
1. Brief Description: A single sentence describing the main objects, their colors, and general positions.
2. Concrete Object Forms: Additional details about the shapes of distinct objects, their specific locations, 
   and geometric attributes.
**Keep the caption short and to the point.** Do not add any extra comments or explanations.

**Examples:**

- **Example 1:**
  
  *Image:* ![Image 1]
  
  **Output:** 
  Green tree on the left and red bench in the center. The tree has a circular canopy and the bench features rectangular slats.

- **Example 2:**
  
  *Image:* ![Image 2]
  
  **Output:** 
  Blue bicycle parked beside the white fence. The bicycle has a triangular frame and the fence consists of vertical rectangular panels.

**Guidelines:**

- **Be Concise:** The caption should be brief, ideally two short sentences as illustrated in the examples.
- **Include Key Details:** Mention the primary objects, their colors, and general locations within the image in the first part.
- **Add Shape Details:** In the second part, provide specific information about the shapes of prominent objects 
  and their precise locations.
- **Avoid Unnecessary Information:** Do not include background context or subjective remarks.
- **Ensure Clarity:** The caption should clearly and accurately describe the visual elements without ambiguity.

**Your Task:**
'''

    conversation_template = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt_text},
                {"type": "text", "text": "*Image:*"},
                {"type": "image"},
                {"type": "text", "text": "Output:"},
            ],
        }
    ]

    # ----------------------------------------------------------------------
    # Helper function to process a single batch
    # ----------------------------------------------------------------------
    def process_batch(batch):
        """
        Given a list of (svg_name, svg_content), convert them to images, 
        run batched inference, and return results (svg_path, caption).
        """
        images = []
        text_prompts = []
        valid_indices = []

        # Convert each SVG to image & build the text prompt
        for idx, (name, content) in enumerate(batch):
            img = load_image_from_svg_content(content)
            if img:
                images.append(img)
                prompt_data = processor.apply_chat_template(conversation_template, add_generation_prompt=True)
                text_prompts.append(prompt_data)
                valid_indices.append(idx)
            else:
                logging.warning(f"Skipping {name} due to image conversion failure.")

        # If no valid images in this batch, just return empty
        if not images:
            return []

        # Prepare inputs in a single batch
        inputs = processor(
            text=text_prompts,
            images=images,
            padding=True,
            return_tensors="pt"
        ).to(model.device)

        # Generate the model outputs in one forward pass
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output_ids = model.generate(
                    **inputs,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.95,
                    max_new_tokens=128,
                )

        # Decode the outputs
        batch_outputs = processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        # Collect results
        results = []
        for local_idx, out_txt in zip(valid_indices, batch_outputs):
            # Remove prefix if present
            if "Output:" in out_txt:
                # split on the known pattern
                parts = out_txt.split("Output:\nassistant\n", 1)
                if len(parts) > 1:
                    out_txt = parts[-1].strip()
            svg_name, svg_contents = batch[local_idx]
            results.append({
                "svg_path": svg_name,
                "caption": out_txt
            })
        return results

    # ----------------------------------------------------------------------
    # 4) Iterate through dataset in batches, write results to CSV
    # ----------------------------------------------------------------------
    logging.info(f"Processing dataset in batches of size {batch_size} and saving to {args.output_csv}...")
    
    with open(args.output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["svg_path", "caption"])
        writer.writeheader()

        buffer = []
        for row_idx, row in tqdm(enumerate(hf_dataset)):
            svg_name = row["svg_name"]
            svg_contents = row["svg_contents"]
            buffer.append((svg_name, svg_contents))

            # Once we hit the batch size, process & write to CSV
            if len(buffer) == batch_size:
                results = process_batch(buffer)
                for r in results:
                    writer.writerow(r)
                buffer.clear()

        # Flush any leftover
        if buffer:
            results = process_batch(buffer)
            for r in results:
                writer.writerow(r)
            buffer.clear()

    logging.info("Finished processing and writing captions to CSV.")

    # ----------------------------------------------------------------------
    # 5) Optionally push to a HF repo.
    # ----------------------------------------------------------------------
    if args.hf_repo:
        logging.info(f"Creating/using the repo '{args.hf_repo}' to push CSV...")
        create_repo(args.hf_repo, repo_type="dataset", private=True, exist_ok=True)
        
        api = HfApi()
        api.upload_file(
            path_or_fileobj=args.output_csv,
            path_in_repo=args.output_csv,
            repo_id=args.hf_repo,
            repo_type="dataset"
        )
        logging.info(f"'{args.output_csv}' pushed to HF repo '{args.hf_repo}'")

    logging.info("All done!")


if __name__ == "__main__":
    main()
