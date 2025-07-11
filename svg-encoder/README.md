# SVG_Encoder

A project for training SVG encoders based on [ModernBERT](https://github.com/AnswerDotAI/ModernBERT).

## Training Process

The SVG encoder model was trained in original ModernBERT settings on a large SVG corpus:
- First stage: 12 billion tokens
- Second stage: 3 billion tokens 
- Third stage: 2 billion tokens

## Usage (current scripts)

Convert trained models to Hugging Face format:

```bash
python scripts/convert_to_hf.py \
  --config path/to/config \
  --checkpoint path/to/checkpoint \
  --tokenizer path/to/tokenizer \
  --output path/to/save \
  --save-tokenizer
```

## Acknowledgements

- [ModernBERT](https://github.com/AnswerDotAI/ModernBERT) for providing the base architecture