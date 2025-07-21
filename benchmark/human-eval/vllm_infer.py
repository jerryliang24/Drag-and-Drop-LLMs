import fire
import torch
from human_eval.data import read_problems, write_jsonl
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

CONFIG_PATH = "../../configs/Qwen1.5"
TEST_ROOT = "./test_ckpts"


# 读取 HumanEval 数据集中的问题
problems = read_problems()


def vllm_human_eval(
    model_name_or_path: str,
    adapter_name_or_path: str = None,
    save_name: str = "generated_predictions.jsonl",
):
    r"""
    Performs batch generation using vLLM engine, which supports tensor parallelism.
    """

    engine_args = {
        "model": model_name_or_path,
        "trust_remote_code": True,
        "dtype": torch.bfloat16,
        "tensor_parallel_size": torch.cuda.device_count(),
        "pipeline_parallel_size": 1,
        "disable_log_stats": True,
        "enable_lora": adapter_name_or_path is not None,
        "gpu_memory_utilization": 0.75,
    }

    num_samples_per_task = 20

    sampling_params = SamplingParams(
        n=num_samples_per_task,
        max_tokens=2000,
        temperature=0.2,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop="###",
    )

    # if is_baseline:
    #         lora_request = LoRARequest("default", 1, adapter_name_or_path)
    lora_request = LoRARequest("default", 1, adapter_name_or_path)

    # lora_request = LoRARequest("default", 1, adapter_name_or_path)

    inputs = [problems[task_id]["prompt"] for task_id in problems]
    results = LLM(**engine_args).generate(inputs, sampling_params, lora_request=lora_request)

    task_ids = [task_id for task_id in problems]
    outputs = [None for _ in task_ids]

    for index, result in zip(range(len(task_ids)), results):
        outputs[index] = [o.text for o in result.outputs]

    samples = []
    for task_id, output in zip(task_ids, outputs):
        for text in output:
            sample = dict(task_id=task_id, completion=text)
            samples.append(sample)

    write_jsonl(save_name, samples)


if __name__ == "__main__":
    fire.Fire(vllm_human_eval)
