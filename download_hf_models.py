from huggingface_hub import snapshot_download
import shutil
import os

# Configuration
repo_id = "ManarAli/babylm-conf"
local_dir = "./fine_tuned_models/babylm-conf"

# Ensure clean destination directory
if os.path.exists(local_dir):
    print(f"⚠️ Removing existing directory: {local_dir}")
    shutil.rmtree(local_dir)

# Download all files from the repo
print(f"⬇️ Downloading {repo_id} to {local_dir}")
snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)

print(f"✅ Download complete. Model saved at: {local_dir}")
