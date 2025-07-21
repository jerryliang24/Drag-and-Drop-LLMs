import subprocess
from typing import List

from fire import Fire


def download_data(lora_types: List[str]):
    """Download the data for the Drag-and-Drop-LLMs project for multiple LoRA types."""
    accept_lora_types = ["common_sense_reasoning", "math1.5B", "coding1.5B", "math7B", "coding7B", "vllm3B"]

    if lora_types == ["all"]:
        lora_types = accept_lora_types
    else:
        invalid = [t for t in lora_types if t not in accept_lora_types]
        assert not invalid, f"Invalid lora_types: {invalid}. Choose from {accept_lora_types}."

    for lora_type in lora_types:
        print(f"\n🚀 Downloading data for lora_type: {lora_type}")
        subprocess.run(
            [
                "huggingface-cli",
                "download",
                f"Jerrylz/{lora_type}",
                "--repo-type",
                "dataset",
                "--local-dir",
                f"./data/{lora_type}",
            ],
            check=True,
        )


if __name__ == "__main__":
    Fire(download_data)
