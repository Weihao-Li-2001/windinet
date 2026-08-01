import os

from huggingface_hub import snapshot_download

local_dir = os.path.join(os.environ["SCRATCH"], "windinet", "euler_mq_dataset")
os.makedirs(local_dir, exist_ok=True)

snapshot_download(
    repo_id="rha6696/euler_mq",
    repo_type="dataset",
    allow_patterns="128x128_ds/*",
    local_dir=local_dir,
)
