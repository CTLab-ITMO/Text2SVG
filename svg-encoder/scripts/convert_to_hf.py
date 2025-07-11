# transformers >= 4.48.0
from transformers import ModernBertModel, ModernBertForMaskedLM
from transformers import ModernBertConfig
import torch
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Convert model checkpoint to Hugging Face format')
    parser.add_argument('--config', required=True, help='Path to the model config file')
    parser.add_argument('--checkpoint', required=True, help='Path to the model checkpoint file')
    parser.add_argument('--tokenizer', required=True, help='Path to the tokenizer')
    parser.add_argument('--output', required=True, help='Path to save the converted model')
    parser.add_argument('--save-tokenizer', action='store_true', help='Save tokenizer with model')
    return parser.parse_args()

args = parse_args()
config_path = args.config
checkpoint_path = args.checkpoint
tokenizer_path = args.tokenizer
save_path = args.output

# Optionally, to save tokenizer with model
do_save_tokenizer = args.save_tokenizer

config = ModernBertConfig.from_pretrained(config_path)
model = ModernBertForMaskedLM(config)

state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
state_dict = state_dict['state']['model']

def strip_prefix(k: str) -> str:
    if 'bert.encoder.' in k:
        return k.replace('bert.encoder.', '')
    elif '.bert.embeddings.' in k:
        return k.replace('bert.', '')
    elif k == "model.bert.final_norm.weight":
        return "model.final_norm.weight"
    elif k == "model.head.dense.weight":
        return "head.dense.weight"
    elif k == "model.head.norm.weight":
        return "head.norm.weight"
    elif k == "model.decoder.weight":
        return "decoder.weight"
    elif k == "model.decoder.bias":
        return "decoder.bias"
    return k

updated_state_dict = {strip_prefix(k): v for k, v in state_dict.items()}

model.load_state_dict(updated_state_dict, strict=True)
model.save_pretrained(save_path)

if do_save_tokenizer:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.save_pretrained(save_path)