

cd ./prepare

CUDA_VISIBLE_DEVICES=0,1 python scripts/vllm_infer.py \
 --model_name_or_path ./models/Qwen2.5-0.5B-Instruct \
 --save_name ./results/ablations/fullshot/fullshot_50.jsonl \
 --dataset ARC-c_test \
 --adapter_name_or_path saves/fullshot/checkpoint-50 \

CUDA_VISIBLE_DEVICES=0,1 python scripts/vllm_infer.py \
 --model_name_or_path ./models/Qwen2.5-0.5B-Instruct \
 --save_name ./results/ablations/fullshot/fullshot_300.jsonl \
 --dataset ARC-c_test \
 --adapter_name_or_path saves/fullshot/checkpoint-300 \
