cd ./workspace/main
bash launch_multi.sh ablation/different_extractors/qwen0.5lora_T5_extractor.py 4

python ablation/different_extractors/qwen0.5lora_generation_for_T5.py --eval_dataset extractor_T51000 --test_dataset ARC-c
python ablation/different_extractors/qwen0.5lora_generation_for_T5.py --eval_dataset extractor_T52000 --test_dataset ARC-c
python ablation/different_extractors/qwen0.5lora_generation_for_T5.py --eval_dataset extractor_T53000 --test_dataset ARC-c
