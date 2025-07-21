cd ./workspace/main
bash launch_multi.sh tasks/common_sense_reasoning/train_qwen0.5lora_ARC-e.py 4

python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset fullARC-eBERT1000 --test_dataset ARC-e
python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset fullARC-eBERT2000 --test_dataset ARC-e
