cd ./workspace/main
bash launch_multi.sh tasks/common_sense_reasoning/train_qwen0.5lora_PIQA.py 4

python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset fullPIQABERT200 --test_dataset PIQA
python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset fullPIQABERT400 --test_dataset PIQA
python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset fullPIQABERT600 --test_dataset PIQA
python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset fullPIQABERT800 --test_dataset PIQA
python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset fullPIQABERT1000 --test_dataset PIQA
