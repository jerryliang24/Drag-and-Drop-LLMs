cd ./workspace/main

python generate/qwen0.5lora_generation_cross.py --eval_dataset fullARC-cBERT1000 --test_dataset science-dataset
python generate/qwen0.5lora_generation_cross.py --eval_dataset fullARC-cBERT2000 --test_dataset science-dataset
python generate/qwen0.5lora_generation_cross.py --eval_dataset fullARC-cBERT3000 --test_dataset science-dataset
