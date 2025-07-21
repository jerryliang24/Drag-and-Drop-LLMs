import argparse

import torch
import transformers
from peft import PeftModel
from transformers import AutoConfig, AutoProcessor


def main():
    args = parse_args()
    lora_model_path = args.lora_model_path
    base_model_name = args.base_model_name
    output_dir = args.output_dir

    processor = AutoProcessor.from_pretrained(base_model_name)

    config = AutoConfig.from_pretrained(base_model_name)
    architecture = config.architectures[0]
    architecture = getattr(transformers, architecture)
    lora_model = PeftModel.from_pretrained(
        architecture.from_pretrained(base_model_name, torch_dtype=torch.bfloat16), lora_model_path, is_trainable=False
    )

    merged_model = lora_model.merge_and_unload()

    processor.save_pretrained(output_dir)
    merged_model.save_pretrained(output_dir)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora_model_path", type=str, default="lora_path")
    parser.add_argument("--base_model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--output_dir", type=str, default="save_path")
    return parser.parse_args()


if __name__ == "__main__":
    main()
