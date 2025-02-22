from unsloth import FastLanguageModel
import torch

checkpoint_path = ""

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = checkpoint_path,
    max_seq_length = 4096,
    dtype = None,
    load_in_4bit = True,
)

#model.push_to_hub('TatonkaHF/Qwen2.5-Coder-32B-Instruct__SVG_v0.2', exists_ok=True)

inf_prompt = """<|im_start|>system
You are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>
<|im_start|>user
Generate SVG image based on the input caption.
Caption: {}<|im_end|>
<|im_start|>assistant
{}"""

# alpaca_prompt = Copied from above
FastLanguageModel.for_inference(model) # Enable native 2x faster inference
inputs = tokenizer(
[
    inf_prompt.format(
        "Rainy day in the city. Detailed image.", # input
        "", # output - leave this blank for generation!
    )
], return_tensors = "pt").to("cuda")

from transformers import TextStreamer
text_streamer = TextStreamer(tokenizer)
_ = model.generate(
    **inputs,
    streamer=text_streamer,
    max_new_tokens=10000,
    temperature=0.3,
    top_p=0.95,
    min_p=0.05,
    top_k=40
)
