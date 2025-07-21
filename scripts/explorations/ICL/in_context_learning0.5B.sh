

cd ./prepare

CUDA_VISIBLE_DEVICES=0,1 python scripts/ICL_test.py \
 --model_name_or_path ./models/Qwen2.5-0.5B-Instruct \
 --save_name ./results/ablations/ICL/1shot.jsonl \
 --max_samples 1 \

CUDA_VISIBLE_DEVICES=0,1 python scripts/ICL_test.py \
 --model_name_or_path ./models/Qwen2.5-0.5B-Instruct \
 --save_name ./results/ablations/ICL/16shot.jsonl \
 --max_samples 16 \

CUDA_VISIBLE_DEVICES=0,1 python scripts/ICL_test.py \
 --model_name_or_path ./models/Qwen2.5-0.5B-Instruct \
 --save_name ./results/ablations/ICL/64shot.jsonl \
 --max_samples 64 \

CUDA_VISIBLE_DEVICES=0,1 python scripts/ICL_test.py \
 --model_name_or_path ./models/Qwen2.5-0.5B-Instruct \
 --save_name ./results/ablations/ICL/256shot.jsonl \
 --max_samples 256 \
