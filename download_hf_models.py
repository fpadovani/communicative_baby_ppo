from huggingface_hub import hf_hub_download
import os

# Specify your model repo and target subfolder
repo_id = "fpadovani/rfscore-kl"
subfolder = "checkpoint-5000"

# Files in that subfolder (manually list them or scrape the repo API)
files = [
    "config.json",
    "generation_config.json",
    "model.safetensors",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json"
]

# Where to store locally
local_dir = "models/rfscore-kl/checkpoint-5000"
os.makedirs(local_dir, exist_ok=True)

# Download each file
for file_name in files:
    hf_hub_download(
        repo_id=repo_id,
        filename=file_name,
        subfolder=subfolder,
        local_dir=local_dir,
        local_dir_use_symlinks=False  # Optional, deprecated
    )
