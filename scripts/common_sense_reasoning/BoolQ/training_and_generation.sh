cd ./workspace/main
bash launch_multi.sh tasks/common_sense_reasoning/train_qwen0.5lora_BoolQ.py 4

python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset fullBoolQBERT1000 --test_dataset BoolQ
python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset fullBoolQBERT2000 --test_dataset BoolQ
python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset fullBoolQBERT3000 --test_dataset BoolQ
