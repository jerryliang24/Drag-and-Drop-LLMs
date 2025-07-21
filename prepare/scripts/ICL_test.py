# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
from typing import Optional

import fire
from llamafactory.extras.misc import check_version, get_device_count
from llamafactory.hparams import get_infer_args
from vllm import LLM, SamplingParams


def vllm_infer(
    model_name_or_path: str,
    template: str = "default",
    cutoff_len: int = 2048,
    max_samples: int = None,
    vllm_config: str = "{}",
    save_name: str = "generated_predictions.jsonl",
    gpu_memory_utilization: float = 0.9,
    temperature: float = 0.95,
    top_p: float = 0.7,
    top_k: int = 50,
    max_new_tokens: int = 1024,
    repetition_penalty: float = 1.0,
    seed: Optional[int] = None,
    pipeline_parallel_size: int = 1,
):
    r"""
    Performs batch generation using vLLM engine, which supports tensor parallelism.
    Usage: python vllm_infer.py --model_name_or_path meta-llama/Llama-2-7b-hf --template llama --dataset alpaca_en_demo
    """
    check_version("vllm>=0.4.3,<=0.7.2")
    if pipeline_parallel_size > get_device_count():
        raise ValueError("Pipeline parallel size should be smaller than the number of gpus.")

    cutoff_len = max_samples * 200

    model_args, data_args, _, generating_args = get_infer_args(
        dict(
            model_name_or_path=model_name_or_path,
            template=template,
            cutoff_len=cutoff_len,
            max_samples=max_samples,
            preprocessing_num_workers=16,
            vllm_config=vllm_config,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
            repetition_penalty=repetition_penalty,
        )
    )

    training_data = json.load(
        open(
            "/fs-computility/ai4sData/zhouyuhao.p/archive/qwengen/OCR_prepare/data/ARC-c_train.json",
            "r",
            encoding="utf-8",
        )
    )

    ICL_prior = [d["prompt"] + "answer: " + d["response"] for d in training_data[:max_samples]]
    ICL_prior = "\n".join(ICL_prior)

    data = json.load(
        open(
            "/fs-computility/ai4sData/zhouyuhao.p/archive/qwengen/OCR_prepare/data/ARC-c_test.json",
            "r",
            encoding="utf-8",
        )
    )

    inputs, labels = [], []
    for i, data_point in enumerate(data):
        prompt = ICL_prior + data_point["prompt"]
        labels.append(data_point["response"])
        inputs.append(prompt)

    sampling_params = SamplingParams(
        repetition_penalty=generating_args.repetition_penalty or 1.0,  # repetition_penalty must > 0
        temperature=generating_args.temperature,
        top_p=generating_args.top_p or 1.0,  # top_p must > 0
        top_k=generating_args.top_k,
        max_tokens=generating_args.max_new_tokens,
        skip_special_tokens=False,
        seed=seed,
    )

    engine_args = {
        "model": model_args.model_name_or_path,
        "trust_remote_code": True,
        "dtype": model_args.infer_dtype,
        "tensor_parallel_size": (get_device_count() // pipeline_parallel_size) or 1,
        "pipeline_parallel_size": pipeline_parallel_size,
        "disable_log_stats": True,
        "enable_lora": model_args.adapter_name_or_path is not None,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_lora_rank": 64,
    }

    if isinstance(model_args.vllm_config, dict):
        engine_args.update(model_args.vllm_config)

    results = LLM(**engine_args).generate(inputs, sampling_params)
    preds = [result.outputs[0].text for result in results]
    import os

    root_dir = save_name.split(os.path.basename(save_name))[0]
    os.makedirs(root_dir, exist_ok=True)
    with open(save_name, "w", encoding="utf-8") as f:
        for pred, label in zip(preds, labels):
            f.write(json.dumps({"predict": pred, "label": label}, ensure_ascii=False) + "\n")

    print("*" * 70)
    print(f"{len(labels)} generated results have been saved at {save_name}.")
    print("*" * 70)


if __name__ == "__main__":
    fire.Fire(vllm_infer)
