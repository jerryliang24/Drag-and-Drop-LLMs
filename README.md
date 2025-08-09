

<h1 align="center"> <img src="./assets/DnD.png" alt="Logo" width="36" /> Drag-and-Drop LLMs: Zero-Shot Prompt-to-Weights</h1>

<div align="center">
<a href='https://jerryliang24.github.io/DnD/' style="text-decoration: none;"><img src='https://img.shields.io/badge/DnD-Projectpage-orange?style=flat&logo=googlehome&logoColor=%23FFFFFF'></a>
<a href='https://arxiv.org/pdf/2506.16406'><img src='https://img.shields.io/badge/arXiv-2506.16406-%23B31B1B?logo=arxiv'></a>
<a href='https://huggingface.co/spaces/Jerrylz/Drag-and-Drop-LLMs'><img src='https://img.shields.io/badge/Hugging%20Face-Demo-blue?style=flat&logo=huggingface&logoColor=%23FFD21E'></a>
<a href='https://huggingface.co/datasets/Jerrylz'><img src='https://img.shields.io/badge/Hugging%20Face-Datasets-blue?style=flat&logo=huggingface&logoColor=%23FFD21E'></a>
<a href='LICENSE'><img src='https://img.shields.io/badge/License-Apache_2.0-green.svg'></a>
<a href='[LICENSE](https://x.com/_akhaliq/status/1937017302999851124)'><img src='https://img.shields.io/twitter/url?url=https%3A%2F%2Fx.com%2F_akhaliq%2Fstatus%2F1937017302999851124
'></a>
</div>

## 🧭 Contents

- 🎥 [Quick Demo](#-customize-your-llms-wo-training-in-seconds)
- 📖 [Abstract](#abstract)
- 🚀 [Installation](#installation)
- 🗂️ [Quick Start](#quick-start)
- 🤖 [Advanced Usage](#advanced-usage)
- 👩‍👩‍👧‍👦 [Acknowledgement](#acknowledgment)
- 📄 [License](#license)
- 🎓 [Citation](#citation)


## 🎥 Customize Your LLMs w/o Training in seconds!


https://github.com/user-attachments/assets/ec1ea0d1-3e1c-47b7-8c30-3623866d9369


**Explore generating your LLMs for various tasks using our [demo](https://huggingface.co/Jerrylz/Drag-and-Drop-LLMs)!**


## 📖 Abstract
Modern Parameter-Efficient Fine-Tuning (PEFT) methods such as low-rank adaptation (LoRA) reduce the cost of customizing large language models (LLMs), yet
 still require a separate optimization run for every downstream dataset. We introduce Drag-and-Drop LLMs (DnD), a prompt-conditioned parameter generator
 that eliminates per-task training by mapping a handful of unlabeled task prompts
 directly to LoRA weight updates. A lightweight text encoder distills each prompt
 batch into condition embeddings, which are then transformed by a cascaded hyper
convolutional decoder into the full set of LoRA matrices. Once trained in a diverse
 collection of prompt-checkpoint pairs, DnD produces task-specific parameters in
 seconds, yielding **i)** up to 12,000× lower overhead than full fine-tuning, **ii)** average
 gains up to 30% in performance over the strongest training LoRAs on unseen
 common-sense reasoning, math, coding, and multimodal benchmarks, and **iii)**
 robust cross-domain generalization despite never seeing the target data or labels.
 Our results demonstrate that prompt-conditioned parameter generation is a viable
 alternative to gradient-based adaptation for rapidly specializing LLMs.


## 🚀 Installation
Before you get started, you need to set up a conda environment first.
1. Create your conda environment.
```shell
conda create -n dnd python=3.12
conda activate dnd
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
```
2. Install dependencies for DnD.

```shell
git clone https://github.com/jerryliang24/Drag-and-Drop-LLMs.git
cd Drag-and-Drop-LLMs
bash install.sh
```


## 🗂️ Quick Start
This section covers the entire process from preparing the checkpoint dataset to training and testing the DnD model.

1. Download foundation models and sentenceBERT in the ./model folder.

```shell
huggingface-cli download <model_name> --local-dir models/<model_name>
# The models you may need for DnD: Qwen/Qwen2.5-0.5/1.5/7B-Instruct, Qwen/Qwen2.5-VL-3B-Instruct, sentence-transformers/all-MiniLM-L12-v2, google-t5/t5-base
```

3. Preparing the checkpoint is laborious, so we recommend using our released LoRA adapters for training.

```shell
python download_data.py --lora_type <lora_type>
# please refer to this script to specify the kind of LoRAs you want to download.
```

If you want to enjoy the process of checkpoint collection, see the [Adapt your own dataset](#adapt-your-own-dataset) section for detailed instructions.

4. Register whether you want to use wandb and your wandb key in [./workspace/main/config.json](https://github.com/jerryliang24/Drag-and-Drop-LLMs/blob/main/workspace/main/config.json).

```shell
{"use_wandb": <bool>,
"wandb_api_key":""}
```


5. Use our example scripts to train DnD model for common sense reasoning dataset ARC-c.
```shell
bash scripts/common_sense_reasoning/ARC-c/training_and_generation.sh
```

You can refer to [./scripts](https://github.com/jerryliang24/Drag-and-Drop-LLMs/tree/main/scripts) folder for a variety of experiments, generating LLMs for common sense reasoning, coding, math, and multimodal tasks.


## 🤖 Advanced Usage


### Reproduce Section 3.4 & Section 3.5: _Ablation Studies and Explorations_

#### condition forms ablation
```shell
bash scripts/ablations/condition_forms/training.sh
bash scripts/ablations/condition_forms/generation.sh
```
This experiment explores different forms of conditions' (Prompts, Prompts+Answers, Answers) influence on DnD's performance.

#### condition extractor ablation

```shell
bash scripts/ablations/extractor_types/<type>_extractor.sh
```

This experiment explores different condition extractors' (Word2Vector, Encoder-Decoder, Decoder-Only) influence on DnD's performance. Note that you need to first download the model in ./models folder.

We also open-source other experiments' code in [./scripts/ablations](https://github.com/jerryliang24/Drag-and-Drop-LLMs/tree/main/scripts/ablations) folder, please feel free to explore!


### Adapt your own dataset
In this section, we will introduce how to train DnD on customized data-checkpoint pairs and further predict novel parameters.

<details>
<summary>Click here for details</summary>


1. Register the dataset

You first need to place your dataset file in [./prepare/data](https://github.com/jerryliang24/Drag-and-Drop-LLMs/tree/main/prepare/data) folder in .json format, and register it in  [./prepare/data/dataset_info.json](https://github.com/jerryliang24/Drag-and-Drop-LLMs/blob/main/prepare/data/dataset_info.json):

```shell
<dataset_name>:
{
  "file_name": "<dataset_name>.json",
  "columns": {"prompt":"prompt",
  "response":"response",
  "system":"system"},
```

Note that the format of your json file should be like:
```
[{ "prompt": "",
  "response": "",
  "system": ""},
  ...,
  ...
  ...,
  { "prompt": "",
  "response": "",
  "system": ""}]
```
  please refer to [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) for more details.


2. Collect checkpoints for this dataset. You need to train LLMs on previous datasets to collect relevent checkpoints, forming data-checkpoint pairs for DnD's training.

We give an example of how to modify the training script:

```diff

### model
-model_name_or_path: ../models/Qwen2.5-0.5B-Instruct
+model_name_or_path: ../models/<your desired base model>


################# line9-21 of training_your_dataset.yaml #################

-lora_rank: 8
+lora_rank: <expected rank>
lora_target: all

### dataset
-dataset: ARC-c
+dataset: <dataset_name> # should be consistent with your json file name
template: qwen
cutoff_len: 2048
-max_samples: 5000
+max_samples: <expected sample>
overwrite_cache: true
preprocessing_num_workers: 16

### output
-output_dir: saves/common_sense_reasoning/ARC-c
+output_dir: saves/<task_name>/<dataset_name>

################# line9-21 of training_your_dataset.yaml #################



################# line28-33 of training_your_dataset.yaml #################
-per_device_train_batch_size: 1
-gradient_accumulation_steps: 8
-learning_rate: 1.0e-4
-num_train_epochs: 1.0
-lr_scheduler_type: cosine
-warmup_ratio: 0.01
#you can modify the training settings
+per_device_train_batch_size:
+gradient_accumulation_steps:
+learning_rate:
+num_train_epochs:
+lr_scheduler_type:
+warmup_ratio:

################# line28-33 of training_your_dataset.yaml #################
```
- After training, you need to do the following to get checkpoint collections.
  1. You need to observe the loss curve, and decide the starting point of fine-tuning for checkpoint collection.
  2. The trainer_state.json in the checkpoint folder (usually named checkpoint-xxx) needs to be modified, setting "save_steps"=1.
  3. You can follow the scripts in [./prepare/training_scripts](https://github.com/jerryliang24/Drag-and-Drop-LLMs/tree/main/prepare/training_scripts) folder that end with "finetune" to design your fine-tuning process.
  4. After running the scripts and obtaining multiple checkpoints, you can simply run [./workspace/datasets/process_datasets/post_process_ckpts.py](https://github.com/jerryliang24/Drag-and-Drop-LLMs/blob/main/workspace/datasets/process_datasets/post_process_ckpts.py) to clean your checkpoint folder, deleting config files and rename checkpoints to ease the process of data loading.


3. Calculate importance scores for the collected checkpoints.

DnD utilizes a weighted MSE for training, assigning different importance to different layers' weights. The specific importance is calculated by the channel-wise variance and we provide scripts in [./workspace/datasets](https://github.com/jerryliang24/Drag-and-Drop-LLMs/tree/main/workspace/datasets), like : criterion_weight_for_<model_type>.py. You need to select a script and adjust it accordingly.

```diff

######################## on line 26-28 in ...<dataset_name>.py ########################
-DATASET_ROOT = "./data/common_sense_reasoning"
-CONFIG_ROOT = f"./workspace/datasets/common_sense_reasoning"
+DATASET_ROOT = "./data/<task_name>"
+CONFIG_ROOT = f"./workspace/datasets/<task_name>"
######################## on line 26-28 in ...<dataset_name>.py ########################



###################### on line 24 in ...<dataset_name>.py #######################

-dataset_tag = "ARC-c"
+dataset_tag = <your dataset_tag>

###################### on line 24 in ...<dataset_name>.py #######################




###################### on line 37 in ...<dataset_name>.py #######################

-datasets = ["ARC-e","OBQA","BoolQ","WinoGrande","PIQA","HellaSwag"]
+datasets = ["<dataset_name_1>","<dataset_name_2>",...,"<dataset_name_n>"]
# All datasets you collect for the target task

###################### on line 37 in ...<dataset_name>.py #######################

4. Create your training script. (<dataset_name> is decided by yourself. And we strongly recommend keeping this name in data registration, checkpoint collection, and DnD training consistent, since it can save much trouble.)

We use ./workspace/main/tasks/common_sense_reasoning/train_qwen0.5lora_ARC-c.py to give an example. You need to create your training script like ./workspace/main/tasks/<task_name>/train_<model_type>_<dataset_name>.py:


```diff

######################## on line 26-28 in ...<dataset_name>.py ########################
-DATASET_ROOT = "./data/common_sense_reasoning"
-CONFIG_ROOT = f"./workspace/datasets/common_sense_reasoning"
+DATASET_ROOT = "./data/<task_name>"
+CONFIG_ROOT = f"./workspace/datasets/<task_name>"
######################## on line 26-28 in ...<dataset_name>.py ########################



###################### on line 24 in ...<dataset_name>.py #######################

-dataset_tag = "ARC-c"
+dataset_tag = <your dataset_tag>

###################### on line 24 in ...<dataset_name>.py #######################




###################### on line 37 in ...<dataset_name>.py #######################

-datasets = ["ARC-e","OBQA","BoolQ","WinoGrande","PIQA","HellaSwag"]
+datasets = ["<dataset_name_1>","<dataset_name_2>",...,"<dataset_name_n>"]
# All datasets you collect for the target task

###################### on line 37 in ...<dataset_name>.py #######################




###################### on line 42-90 in ...<dataset_name>.py #######################
  config: dict[str, [float, int, str, dict]] = {
    # global setting
    "seed": SEED,
    "model_tag": os.path.basename(__file__)[:-3].split("_")[1],
    "need_test": False,
    "use_wandb": True,
    # data setting
-    "token_size": (8, 128),
+    "token_size": <suitable token size>
-    "real_length": 50,
+    "real_length": <number of checkpoints you like to use>
    "train_checkpoint_folders": [f"{DATASET_ROOT}/{dataset}" for dataset in datasets],
    "test_checkpoint_folder": "",
    "dataset_tag": dataset_tag,
    "generated_file": f"{CONFIG_ROOT}/{dataset_tag}/",
    # train setting
    "max_num_gpus": 8,
-    "batch_size": 64,
+    "batch_size": <suitable batch_size>
-    "num_workers": 8,
+    "num_workers": <suitable num_workers>
    "prefetch_factor": 1,
    "warmup_steps": 1,
-    "total_steps": 4000,
-    "learning_rate": 3e-5,
+    "total_steps": <your preferred training setting>
+    "learning_rate":
    "weight_decay": 0.1,
    "max_grad_norm": 1.0,
    "save_every": 100,
    "print_every": 20,
-    "num_texts": 128,
+    "num_texts": <suitable length of prompt batch>
    "save_folder": "./checkpoints",
    "noise_enhance": 0.0001,
    "criterion_weight": calculate_mean_criterion_weight([f"{CONFIG_ROOT}/{dataset}/criterion_weight.pt" for dataset in datasets]),
    "extractor_type":"BERT",
    "text_tokenizer":AutoTokenizer.from_pretrained(extractor),
    "extra_condition_module":
        AutoModel.from_pretrained(extractor,
        torch_dtype="auto").to(accelerator.device),
    "max_text_length":max_text_length,

-    "model_config": {
-        "features": [
-            (128, max_text_length, 384), (128, 200, 300),
-            (128, 100, 256), (256, 50, 200),
-            (512, 50, 200),
-            (1024, 25, 200), (1024, 10, 200), (2048, 10, 200),
-            (4296, 8, 128),
-        ],
-        "condition_dim": (128, max_text_length, 384),
-        "kernel_size": 9,
-    },
+     <your desired model size>
}
###################### on line 42-90 in ...<dataset_name>.py #######################
```

5. Train DnD model.
```shell
cd ./workspace/main
bash launch_multi.sh tasks/<task_name>/train_<model_type>_<dataset_name>.py <number_of_gpus>
```

Note that the adjustment of generation scripts is similar.

</details>

## 👩‍👩‍👧‍👦 Acknowledgment
We sincerely appreciate
Yuxiang Li,
Jiaxin Wu,
Zhiheng Chen,
Lei Feng,
Jingle Fu,
[Bohan Zhuang](https://bohanzhuang.github.io/),
[Ziheng Qin](https://henryqin1997.github.io/ziheng_qin/),
[Zangwei Zheng](https://zhengzangw.github.io/),
[Zihan Qiu](https://www.linkedin.com/in/zihan-qiu-33a172249/),
[Zexi Li](https://zexilee.github.io/about-zexili//),
[Gongfan Fang](https://fangggf.github.io/),
[Xinyin Ma](https://horseee.github.io/),
and [Qinglin Lu](https://openreview.net/profile?id=~Qinglin_Lu2) for valuable discussions and feedbacks during this work.
<!-- This research is supported by the National Research Foundation,
Singapore under its AI Singapore Programme
(AISG Award No: AISG2-PhD-2021-08-008). -->


## 📄 License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.


## 🎓 Citation
```
@misc{liang2025draganddropllmszeroshotprompttoweights,
      title={Drag-and-Drop LLMs: Zero-Shot Prompt-to-Weights},
      author={Zhiyuan Liang and Dongwen Tang and Yuhao Zhou and Xuanlei Zhao and Mingjia Shi and Wangbo Zhao and Zekai Li and Peihao Wang and Konstantin Schürholt and Damian Borth and Michael M. Bronstein and Yang You and Zhangyang Wang and Kai Wang},
      year={2025},
      eprint={2506.16406},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2506.16406},
}
```

---

[![Star History Chart](https://api.star-history.com/svg?repos=jerryliang24/Drag-and-Drop-LLMs&type=Date)](https://star-history.com/#jerryliang24/Drag-and-Drop-LLMs&Date)

<div align="center">
  <p><strong>🌟 Star us on GitHub if you find DnD helpful! 🌟</strong></p>
</div>
