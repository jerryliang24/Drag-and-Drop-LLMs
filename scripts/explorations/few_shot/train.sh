export CUDA_VISIBLE_DEVICES=0,1,2,3
cd ./prepare

llamafactory-cli training_scripts/ablations/few-shot_finetune/qwen2.5_lora_sft_arcc_fewshot-1.yaml
llamafactory-cli training_scripts/ablations/few-shot_finetune/qwen2.5_lora_sft_arcc_fewshot-16.yaml
llamafactory-cli training_scripts/ablations/few-shot_finetune/qwen2.5_lora_sft_arcc_fewshot-64.yaml
llamafactory-cli training_scripts/ablations/few-shot_finetune/qwen2.5_lora_sft_arcc_fewshot-256.yaml
