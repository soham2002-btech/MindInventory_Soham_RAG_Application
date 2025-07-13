import os
from huggingface_hub import hf_hub_download

# You can change this to any GGUF quantized model you want
# Example: Mistral-7B-Instruct GGUF (Q4_K_M)
REPO_ID = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF"
FILENAME = "mistral-7b-instruct-v0.2.Q4_K_M.gguf"
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Downloading {FILENAME} from {REPO_ID} ...")
model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, local_dir=MODEL_DIR, local_dir_use_symlinks=False)
print(f"Model downloaded to: {model_path}") 