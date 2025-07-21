cd ./workspace/main
bash launch_multi.sh tasks/common_sense_reasoning/train_qwen0.5lora_HellaSwag.py 4

python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset fullHellaSwagBERT400 --test_dataset HellaSwag
python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset fullHellaSwagBERT800 --test_dataset HellaSwag
python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset fullHellaSwagBERT1200 --test_dataset HellaSwag
python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset fullHellaSwagBERT1600 --test_dataset HellaSwag
python generate/qwen0.5lora_generation_common_sense_reasoning.py --eval_dataset fullHellaSwagBERT2000 --test_dataset HellaSwag
