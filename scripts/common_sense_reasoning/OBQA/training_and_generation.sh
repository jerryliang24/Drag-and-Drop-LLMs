cd ./workspace/main
bash launch_multi.sh tasks/common_sense_reasoning/train_qwen0.5lora_OBQA.py 4

python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset OBQA1000 --test_dataset OBQA
python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset OBQA2000 --test_dataset OBQA
python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset OBQA3000 --test_dataset OBQA
