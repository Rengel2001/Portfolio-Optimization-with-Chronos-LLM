import os
from transformers import AutoModelForSeq2SeqLM

# Set the cache directory
hf_cache_dir = "/home/ryengel/.cache/huggingface"

# Set environment variables
os.environ["TRANSFORMERS_CACHE"] = hf_cache_dir
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Chronos-T5 model from Hugging Face
model_path = "amazon/chronos-bolt-base"
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
print(f"Chronos-T5 has been loaded successfully: {model_path}") 