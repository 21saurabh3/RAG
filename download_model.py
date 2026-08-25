from huggingface_hub import snapshot_download

local_path = snapshot_download(
    repo_id="sentence-transformers/all-MiniLM-L6-v2",
    local_dir="./models/all-MiniLM-L6-v2",
)

print(f"Model downloaded to: {local_path}")