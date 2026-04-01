"""
Download Chandra OCR 2 weights into the 'chandra-weights' Modal Volume.

Run once before deploying modal_app.py:
    modal run chandra/setup_volume.py
"""

import modal

volume = modal.Volume.from_name("chandra-weights", create_if_missing=True)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install(
        "huggingface_hub[hf_transfer]>=0.30.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

app = modal.App("chandra-ocr-2-setup")


@app.function(
    image=image,
    volumes={"/model": volume},
    timeout=1800,
)
def download_weights():
    from huggingface_hub import snapshot_download
    import os

    dest = "/model/chandra-ocr-2"
    if os.path.exists(os.path.join(dest, "config.json")):
        print("Weights already present — nothing to do.")
        return

    print("Downloading datalab-to/chandra-ocr-2 ...")
    snapshot_download(
        repo_id="datalab-to/chandra-ocr-2",
        local_dir=dest,
        local_dir_use_symlinks=False,
    )
    volume.commit()
    print("Done.  Weights saved to /model/chandra-ocr-2")


@app.local_entrypoint()
def main():
    download_weights.remote()
