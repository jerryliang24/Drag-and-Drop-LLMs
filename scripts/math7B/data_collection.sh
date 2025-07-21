export CUDA_VISIBLE_DEVICES=0,1,2,3
cd ./prepare

llamafactory-cli train training_scripts/math7B/qwen2.5_lora_sft_Math_QA_finetune.yaml
llamafactory-cli train training_scripts/math7B/qwen2.5_lora_sft_Competition_Math_finetune.yaml
llamafactory-cli train training_scripts/math7B/qwen2.5_lora_sft_Mu-Math_finetune.yaml
llamafactory-cli train training_scripts/math7B/qwen2.5_lora_sft_ToT-Math-V1_finetune.yaml
llamafactory-cli train training_scripts/math7B/qwen2.5_lora_sft_Math-Plus_finetune.yaml
llamafactory-cli train training_scripts/math7B/qwen2.5_lora_sft_Math-IIO-68K-Mini_finetune.yaml
