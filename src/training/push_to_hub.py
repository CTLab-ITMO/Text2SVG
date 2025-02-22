from unsloth import FastLanguageModel
import torch

model_path = ""
repo_path = ""

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = 4096,
    dtype = None,
    load_in_4bit = True,
    device='cpu'
)

model.push_to_hub_merged(repo_path, tokenizer, save_method = "merged_4bit")
