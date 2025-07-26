cd ./workspace/main
bash launch_multi.sh tasks/common_sense_reasoning/train_qwen0.5lora_ARC-c.py 4

export CUDA_VISIBLE_DEVICES=0,1
# Qwen0.5B has 14 attention heads and can only parallel on 2 or 7 GPUs

python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset ARC-c --test_dataset ARC-c
python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset ARC-c --test_dataset ARC-c
python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset ARC-c --test_dataset ARC-c
python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset ARC-c --test_dataset ARC-c
