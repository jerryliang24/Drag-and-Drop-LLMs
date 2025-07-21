cd ./workspace/main
bash launch_multi.sh tasks/common_sense_reasoning/train_qwen0.5lora_ARC-c.py 4

python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset fullARC-cBERT1000 --test_dataset ARC-c
python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset fullARC-cBERT2000 --test_dataset ARC-c
python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset fullARC-cBERT3000 --test_dataset ARC-c
python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset fullARC-cBERT4000 --test_dataset ARC-c
