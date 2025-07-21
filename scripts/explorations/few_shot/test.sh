

cd ./prepare

CUDA_VISIBLE_DEVICES=0,1 python scripts/vllm_infer.py \
 --model_name_or_path ./models/Qwen2.5-0.5B-Instruct \
 --save_name ./results/ablations/fewshot/ARC-c_1shot.jsonl \
 --dataset ARC-c_test \
 --adapter_name_or_path saves/few-shot/ARC-c_1shot \

CUDA_VISIBLE_DEVICES=0,1 python scripts/vllm_infer.py \
 --model_name_or_path ./models/Qwen2.5-0.5B-Instruct \
 --save_name ./results/ablations/fewshot/ARC-c_16shot.jsonl \
 --dataset ARC-c_test \
 --adapter_name_or_path saves/few-shot/ARC-c_16shot \

CUDA_VISIBLE_DEVICES=0,1 python scripts/vllm_infer.py \
 --model_name_or_path ./models/Qwen2.5-0.5B-Instruct \
 --save_name ./results/ablations/fewshot/ARC-c_64shot.jsonl \
 --dataset ARC-c_test \
 --adapter_name_or_path saves/few-shot/ARC-c_64shot \


CUDA_VISIBLE_DEVICES=0,1 python scripts/vllm_infer.py \
 --model_name_or_path ./models/Qwen2.5-0.5B-Instruct \
 --save_name ./results/ablations/fewshot/ARC-c_256shot.jsonl \
 --dataset ARC-c_test \
 --adapter_name_or_path saves/few-shot/ARC-c_256shot \
