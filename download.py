import os
import httpx
from huggingface_hub import snapshot_download
from huggingface_hub.utils import get_session

os.environ["HF_HUB_DISABLE_SSL_VERIFY"] = "1"

# 覆盖httpx session，关闭证书校验
session = get_session()
session.verify = False

snapshot_download(
    repo_id="z-lab/Qwen3-8B-DFlash-b16",
    local_dir="/home/weight/Qwen3-8B-DFlash-b16",
)
