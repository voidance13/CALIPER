import os

os.environ["HF_ENDPOINT"] = "https://huggingface.co"
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "120"

from huggingface_hub import snapshot_download


print(
    snapshot_download(
        "google-t5/t5-large",
        local_dir="/mnt/data/caliper/models/t5-large",
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "model.safetensors",
            "spiece.model",
            "tokenizer.json",
            "tokenizer_config.json",
        ],
    )
)
