# Jupyter cell — Batch SVG caption mining с динамическими промптами

import json, io, random
from pathlib import Path
import datasets
from optimization.rasterization import svg_to_png_bytes
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM, GenerationConfig
from tqdm import tqdm

# ─── ПАРАМЕТРЫ ────────────────────────────────────────────────────────────────
INPUT_JSONL   = "input_paths_hashes_stage_2.jsonl"
OUTPUT_JSONL  = Path("captions_stage_2.jsonl")
RESOLUTION    = 256
BATCH_SIZE    = 12
DEVICE        = "cuda"
SOURCE        = "Phi-4-multimodal-instruct"

# ─── Списки настроек для генерации вопросов ─────────────────────────────────
possible_styles     = ["concise", "detailed", "technical", "schematic"]
possible_focuses    = ["object-focus", "scene-focus", "color-focus", "action-focus"]
possible_max_words  = [2, 5, 10, 20, 50, 100]

def create_question(style: str, focus: str, max_words: int) -> str:
    """
    Возвращает строку вопроса вида:
      'Generate a {style}, {focus} caption no longer than {max_words} words.'
    """
    return (
        f"Generate a {style} caption with {focus}, "
        f"no more than {max_words} words, describing this image."
    )

# ─── Загрузка данных и модели ─────────────────────────────────────────────────
data = datasets.Dataset.from_json(INPUT_JSONL)
N = len(data)
OUTPUT_JSONL.parent.mkdir(parents=True, exist_ok=True)

model_name = "/hdd/Models/Phi-4-multimodal-instruct"
processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    _attn_implementation="flash_attention_2"
)
model.to(DEVICE)
generation_config = GenerationConfig.from_pretrained(model_name, "generation_config.json")

SYSTEM      = "<|system|>You are an expert image captioning assistant.<|end|>"
USER_PREF   = "<|user|><|image_1|>"
ASSIST_PREF = "<|assistant|>"
SUFFIX      = "<|end|>"

# фиксируем сид для воспроизводимости
random.seed(1234)

# ─── Основной цикл по батчам ───────────────────────────────────────────────────
for start in tqdm(range(0, N, BATCH_SIZE), desc="Batches"):
    batch = [data[i] for i in range(start, min(start+BATCH_SIZE, N))]
    images, prompts, metas = [], [], []
    
    for item in batch:
        # 1) рисуем PNG
        svg_path = Path(item["path"])
        try:
            png = svg_to_png_bytes(svg_path.read_bytes(), size=RESOLUTION, timeout=2)
        except Exception:
            continue
        if not png:
            continue
        
        img = Image.open(io.BytesIO(png)).convert("RGB")
        
        # 2) выбираем случайные параметры промпта
        style    = random.choice(possible_styles)
        focus    = random.choice(possible_focuses)
        max_w    = random.choice(possible_max_words)
        question = create_question(style, focus, max_w)
        
        # 3) собираем финальный промпт
        prompt = f"{SYSTEM}{USER_PREF}{question}{SUFFIX}{ASSIST_PREF}"
        
        images.append(img)
        prompts.append(prompt)
        metas.append({
            "path": str(svg_path),
            "style": style,
            "focus": focus,
            "max_words": max_w,
            "question": question
        })
    
    if not images:
        continue
    
    # 4) токенизация + генерация
    inputs = processor(text=prompts, images=images, return_tensors="pt", padding=True).to(DEVICE)
    out_ids = model.generate(
        **inputs,
        generation_config=generation_config,
        max_new_tokens=128,
    )
    out_ids = out_ids[:, inputs["input_ids"].shape[1]:]
    captions = processor.batch_decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    
    # 5) пишем результаты
    with OUTPUT_JSONL.open("a", encoding="utf-8") as fout:
        for meta, cap in zip(metas, captions):
            rec = {
                "path":       meta["path"],
                "caption":    cap.strip(),
                "source":     SOURCE,
                "prompt":     prompt,
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

print("✅ Batch captioning complete.")
