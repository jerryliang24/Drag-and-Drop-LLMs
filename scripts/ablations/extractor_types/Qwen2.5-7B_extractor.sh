cd ./workspace/main
bash launch_multi.sh ablation/different_extractors/qwen0.5lora_Qwen2.5-7B_extractor.py 4

python ablation/different_extractors/qwen0.5lora_generation_for_7B.py --eval_dataset extractor_Qwen1000 --test_dataset ARC-c
python ablation/different_extractors/qwen0.5lora_generation_for_7B.py --eval_dataset extractor_Qwen2000 --test_dataset ARC-c
python ablation/different_extractors/qwen0.5lora_generation_for_7B.py --eval_dataset extractor_Qwen3000 --test_dataset ARC-c
