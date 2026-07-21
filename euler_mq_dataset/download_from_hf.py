from huggingface_hub import snapshot_download 

snapshot_download(
    repo_id="rha6696/euler_mq",
    repo_type="dataset",
    allow_patterns="128x128_ds/*",
    local_dir="."
)
