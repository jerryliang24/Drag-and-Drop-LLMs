import json
import os
import sys

root = os.sep + os.sep.join(__file__.split(os.sep)[1 : __file__.split(os.sep).index("Drag-and-Drop-LLMs") + 1])
sys.path.append(root)
os.chdir(root)
os.environ["NUM_PROCESSES"] = "1"
model_type = os.path.basename(__file__).split("_")[0]

import gc
import shutil
import subprocess

import torch
from fire import Fire
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from workspace.dnd.dataset import Text2Qwen25LoRA_FullCondDataset as Dataset
from workspace.dnd.model import HyperConvDecoderModel_FullCond as Model
from workspace.dnd.tokenizer import Qwen253BVL_LoRA_Tokenizer2D as Tokenizer

SEED = 999
DATASET_ROOT = "./data/vllm3B"
CONFIG_ROOT = "./workspace/datasets/vllm3B"
COND_ROOT = "./prepare/data"
SAVE_ROOT = "./generated/vllm_3B"
extractor = "./models/all-MiniLM-L12-v2"
TEST_ROOT = "./test_ckpts"
CONFIG_PATH = "./configs/Qwen3B"
RES_ROOT = "./results/vllm3B"

import torch

torch.set_float32_matmul_precision("high")
import accelerate.utils
from torch.utils.data import DataLoader

accelerate.utils.set_seed(SEED)
datasets = ["MathV360K"]
dataset_tag = "MathV360K"
max_text_length = 1536
config: dict[str, [float, int, str, dict]] = {
    # global setting
    "need_test": False,
    # data setting
    "token_size": (16, 256),
    "real_length": 10,
    "num_texts": 16,
    "criterion_weight": torch.load(
        f"{CONFIG_ROOT}/{dataset_tag}/criterion_weight.pt", map_location="cpu", weights_only=True
    ),
    "extractor_type": "BERT",
    "text_tokenizer": AutoTokenizer.from_pretrained(extractor),
    "extra_condition_module": AutoModel.from_pretrained(extractor, torch_dtype="auto"),
    "max_text_length": max_text_length,
    "model_config": {
        "features": [
            (16, max_text_length, 384),
            (64, 500, 300),
            (256, 125, 300),
            (1024, 64, 300),
            (2048, 16, 256),
            (7308, 16, 256),
        ],
        "condition_dim": (16, max_text_length, 384),
        "kernel_size": 11,
    },
}


generate_config = {
    "device": "cuda",
    "num_generated": 10,
    "need_test": False,
}
config.update(generate_config)


def find_safetensors_files(directory):
    safetensors_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".safetensors"):
                safetensors_files.append(os.path.join(root, file))
    return safetensors_files


def copy_files(src_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    for root, dirs, files in os.walk(src_dir):
        for file in files:
            src_file_path = os.path.join(root, file)
            relative_path = os.path.relpath(root, src_dir)
            dest_folder_path = os.path.join(dest_dir, relative_path)
            if not os.path.exists(dest_folder_path):
                os.makedirs(dest_folder_path)
            dest_file_path = os.path.join(dest_folder_path, file)
            shutil.copy2(src_file_path, dest_file_path)
            # print(f"files transferred to {dest_file_path} complete")


def process_adapter_path(adapter_dir):
    ckpts = find_safetensors_files(adapter_dir)
    for ckpt in ckpts:
        dir_name = adapter_dir.split("/")[-1] + "_" + os.path.basename(ckpt).split(".")[0]
        save_dir = os.path.join("./test_ckpts", dir_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            copy_files(CONFIG_PATH, save_dir)
            shutil.copy2(ckpt, save_dir)
            new_ada = os.path.join(save_dir, os.path.basename(ckpt))
            os.rename(new_ada, os.path.join(save_dir, "adapter_model.safetensors"))


# generate
print("==> Defining generate..")


def generate(model, loader, dataset, dstag_T, dstag_V):
    print("==> Generating...")
    model.eval()
    # prepare data
    for idx, (tokens, cond_id, cond_mask, tag) in enumerate(loader):
        # generate
        if idx >= config["num_generated"]:
            break
        with torch.no_grad() and torch.autocast("cuda", dtype=torch.bfloat16):
            mask = ~torch.isnan(tokens)
            tokens = torch.nan_to_num_(tokens, nan=0.0)
            conditions = {
                "input_ids": cond_id.to(device=config["device"]),
                "attention_mask": cond_mask.to(device=config["device"]),
            }

            import threading
            import time

            start_time = time.time()
            stop_timer = threading.Event()

            def display_time():
                while not stop_timer.is_set():
                    elapsed_time = time.time() - start_time
                    print(f"Elapsed time: {elapsed_time:.3f} seconds", end="\r")
                    time.sleep(0.05)

            timer_thread = threading.Thread(target=display_time)
            timer_thread.start()

            predict = model(
                source=None,
                mask=mask.to(config["device"]),
                condition=conditions,
                target=None,
                generate=True,
            )  # generate

            stop_timer.set()  # 告诉计时线程停止
            timer_thread.join()
            print()

        print(f"generate the {idx+1} th...")
        # save and log
        generated_norm = torch.square(predict[mask]).mean().item()
        original_norm = torch.square(tokens[mask]).mean().item()
        print("generated_start:", predict.flatten()[0:5].tolist())
        print("original_start:", tokens.flatten()[0:5].tolist())
        print("generated_end:", predict[0, -1, 0, 0:5].tolist())
        print("original_end:", tokens[0, -1, 0, 0:5].tolist())
        print("generated_l2norm:", generated_norm)
        print("original_l2norm:", original_norm)
        # noinspection PyTypeChecker save checkpoint
        save_path = dataset.save_checkpoint(
            save_path=f"{SAVE_ROOT}/{dstag_T}T_on_{dstag_V}V/", tokens=predict[0], tag=tag, number=idx
        )  # save files
        if config["need_test"]:
            os.system(config["test_command"].format(save_path))


def main(eval_dataset: str, test_dataset: str):
    # Model
    print("==> Building model..")
    model = Model(
        config=config["model_config"],
        criterion_weight=config["criterion_weight"].view(1, -1, 1, 1),
        extractor_type=config["extractor_type"],
        extra_condition_module=config["extra_condition_module"],
    )

    tokenizer = Tokenizer(token_size=config["token_size"])
    diction = torch.load(f"./checkpoints/{model_type}__{eval_dataset}.pth", weights_only=True, map_location="cpu")
    model.load_state_dict(diction, strict=False)
    model.to(config["device"])

    # load module
    test_set = Dataset(
        checkpoint_folders=[f"{DATASET_ROOT}/MathV360K"],
        tokenizer=tokenizer,
        expected_iteration=None,
        real_length=config["real_length"],
        texts=[json.load(open(f"{COND_ROOT}/{test_dataset}.json", "r", encoding="utf-8"))],
        num_texts=config["num_texts"],
        text_tokenizer=config["text_tokenizer"],
        max_text_length=config["max_text_length"],
    )  # test_set
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=1,
        num_workers=0,
        collate_fn=test_set.collate_fn_test,
        shuffle=False,
    )  # test dataloader

    print(f"\n\nGenerating parameters for {test_dataset}")
    generate(model, test_loader, test_set, eval_dataset, test_dataset)
    del model, test_loader
    torch.cuda.empty_cache()
    gc.collect()

    process_adapter_path(f"{SAVE_ROOT}/{eval_dataset}T_on_{test_dataset}V")
    print("==> Start testing..")

    for i in range(config["real_length"]):
        args = [
            "--base_model_name",
            "./models/Qwen2.5-VL-3B-Instruct",
            "--output_dir",
            f"{TEST_ROOT}/{eval_dataset}T_on_{test_dataset}V_{i}_merged",
            "--lora_model_path",
            f"{TEST_ROOT}/{eval_dataset}T_on_{test_dataset}V_{i}",
        ]

        subprocess.run(["python", "./benchmark/merge_lora.py"] + args)
        subprocess.run(["python", "./benchmark/VLMEvalkKit/run.py" "--config", "./benchmark/VLMEvalKit/config.json"])

        import shutil

        shutil.rmtree(
            f"{TEST_ROOT}/{eval_dataset}T_on_{test_dataset}V_{i}_merged",
        )


if __name__ == "__main__":
    Fire(main)
