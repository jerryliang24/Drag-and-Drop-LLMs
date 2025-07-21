cd ./workspace/main
bash launch_multi.sh ablation/different_extractors/qwen0.5lora_Glove_extractor.py 4

python ablation/different_extractors/qwen0.5lora_generation_for_Glove.py --eval_dataset extractor_Glove1000 --test_dataset ARC-c
python ablation/different_extractors/qwen0.5lora_generation_for_Glove.py --eval_dataset extractor_Glove2000 --test_dataset ARC-c
python ablation/different_extractors/qwen0.5lora_generation_for_Glove.py --eval_dataset extractor_Glove3000 --test_dataset ARC-c
