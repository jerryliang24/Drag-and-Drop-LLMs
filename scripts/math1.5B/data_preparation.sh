export CUDA_VISIBLE_DEVICES=0,1,2,3
cd ./prepare

llamafactory-cli train training_scripts/math1.5B/qwen2.5_lora_sft_Math_QA_pretrain.yaml
llamafactory-cli train training_scripts/math1.5B/qwen2.5_lora_sft_Competition_Math_pretrain.yaml
llamafactory-cli train training_scripts/math1.5B/qwen2.5_lora_sft_Mu-Math_pretrain.yaml
llamafactory-cli train training_scripts/math1.5B/qwen2.5_lora_sft_ToT-Math-V1_pretrain.yaml
llamafactory-cli train training_scripts/math1.5B/qwen2.5_lora_sft_Math-Plus_pretrain.yaml
llamafactory-cli train training_scripts/math1.5B/qwen2.5_lora_sft_Math-IIO-68K-Mini_pretrain.yaml
