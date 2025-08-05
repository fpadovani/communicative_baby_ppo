from huggingface_hub import snapshot_download
import shutil
import os

repo_id = "ManarAli/babylm-conf"
local_dir = "./fine_tuned_models/babylm-conf"

# Clean download
if os.path.exists(local_dir):
    shutil.rmtree(local_dir)

snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)

