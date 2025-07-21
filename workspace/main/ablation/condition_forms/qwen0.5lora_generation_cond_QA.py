import json
import os
import subprocess
import sys

file_path = os.path.abspath(sys.argv[1])
root = os.sep + os.sep.join(__file__.split(os.sep)[1 : __file__.split(os.sep).index("Drag-and-Drop-LLMs") + 1])
sys.path.append(root)
os.chdir(root)
os.environ["NUM_PROCESSES"] = "1"
model_type = os.path.basename(__file__).split("_")[0]

import gc
import shutil

import torch
from fire import Fire
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

from workspace.dnd.dataset import Text2Qwen25LoRA_CondQ_ADataset as Dataset
from workspace.dnd.model import HyperConvDecoderModel_FullCond as Model
from workspace.dnd.tokenizer import Qwen2505LoRA_Tokenizer2D as Tokenizer

SEED = 999
DATASET_ROOT = "./data/common_sense_reasoning"
CONFIG_ROOT = "./workspace/datasets/common_sense_reasoning"
COND_ROOT = "./prepare/data"
SAVE_ROOT = "./generated/common_sense_reasoning"
extractor = "./models/all-MiniLM-L12-v2"
TEST_ROOT = "../test_ckpts"
CONFIG_PATH = "./configs/Qwen0.5"
RES_ROOT = "../results/ablations/condition_forms"

import torch

torch.set_float32_matmul_precision("high")
import accelerate.utils
from torch.utils.data import DataLoader

accelerate.utils.set_seed(SEED)
max_text_length = 384
dataset_tag = "BoolQ"
config: dict[str, [float, int, str, dict]] = {
    # global setting
    "need_test": False,
    # data setting
    "token_size": (8, 128),
    "real_length": 50,
    "num_texts": 128,
    "criterion_weight": torch.load(
        f"{CONFIG_ROOT}/{dataset_tag}/criterion_weight.pt", map_location="cpu", weights_only=True
    ),
    "extractor_type": "BERT",
    "text_tokenizer": AutoTokenizer.from_pretrained(extractor),
    "extra_condition_module": AutoModel.from_pretrained(extractor, torch_dtype="auto"),
    "max_text_length": 384,
    "model_config": {
        "features": [
            (128, max_text_length, 384),
            (128, 200, 300),
            (128, 100, 256),
            (256, 50, 200),
            (512, 50, 200),
            (1024, 25, 200),
            (1024, 10, 200),
            (2048, 10, 200),
            (4296, 8, 128),
        ],
        "condition_dim": (128, max_text_length, 384),
        "kernel_size": 9,
    },
}
model = Model(
    config=config["model_config"],
    criterion_weight=config["criterion_weight"].view(1, -1, 1, 1),
    extractor_type=config["extractor_type"],
    extra_condition_module=config["extra_condition_module"],
)
tokenizer = Tokenizer(token_size=config["token_size"])


generate_config = {
    "device": "cuda",
    "num_generated": 10,
    "need_test": False,
}
config.update(generate_config)


# generate
print("==> Defining generate..")


def generate(model, loader, dataset, dstag_T, dstag_V):
    print("==> Generating...")
    model.eval()
    # prepare data
    for idx, (tokens, cond_id, cond_mask, tag) in enumerate(loader):
        # generate
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

        print(f"generate the {idx} th...")
        # save and log
        generated_norm = torch.square(predict[mask]).mean().item()
        original_norm = torch.square(tokens[mask]).mean().item()
        print("generated_start:", predict.flatten()[0:5].tolist())
        print("original_start:", tokens.flatten()[0:5].tolist())
        print("generated_end:", predict[0, -1, 0, 0:5].tolist())
        print("original_end:", tokens[0, -1, 0, 0:5].tolist())
        print("generated_l2norm:", generated_norm)
        print("original_l2norm:", original_norm)
        torch.cuda.empty_cache()
        # noinspection PyTypeChecker save checkpoint
        save_path = dataset.save_checkpoint(
            save_path=f"{SAVE_ROOT}/{dstag_T}T_on_{dstag_V}V", tokens=predict[0], tag=tag, number=idx
        )  # save files


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


def main(eval_dataset: str, test_dataset: str):
    # Model
    print("==> Building model..")
    diction = torch.load(f"./checkpoints/{model_type}__{eval_dataset}.pth", weights_only=True, map_location="cpu")
    model.load_state_dict(diction, strict=False)
    model.to(config["device"])

    # load module
    test_set = Dataset(
        checkpoint_folders=[f"{DATASET_ROOT}/{test_dataset}"],
        tokenizer=tokenizer,
        expected_iteration=None,
        real_length=config["real_length"],
        texts=[json.load(open(f"{COND_ROOT}/{test_dataset}_test.json", "r", encoding="utf-8"))],
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
    generate(test_loader, test_set, eval_dataset, test_dataset)
    del model, test_loader
    torch.cuda.empty_cache()
    gc.collect()

    process_adapter_path(f"{SAVE_ROOT}/{eval_dataset}T_on_{test_dataset}V")
    print("==> Start testing..")
    os.chdir("./prepare")

    for i in range(config["real_length"]):
        args = [
            "--model_name_or_path",
            "../models/Qwen2.5-0.5B-Instruct",
            "--save_name",
            f"{RES_ROOT}/{test_dataset}/{eval_dataset}T_on_{test_dataset}V_{i}.jsonl",
            "--dataset",
            f"{test_dataset}_test",
            "--adapter_name_or_path",
            f"{TEST_ROOT}/{eval_dataset}T_on_{test_dataset}V_{i}",
        ]

        subprocess.run(["python", "scripts/vllm_infer.py"] + args)
        subprocess.run(
            ["python", "calculate_acc.py"]
            + ["--file", f"{RES_ROOT}/{test_dataset}/{eval_dataset}T_on_{test_dataset}V_{i}.jsonl"]
        )


if __name__ == "__main__":
    Fire(main)
