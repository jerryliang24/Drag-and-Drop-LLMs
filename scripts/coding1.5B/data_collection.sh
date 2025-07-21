export CUDA_VISIBLE_DEVICES=4,5,6,7

cd ./prepare

llamafactory-cli train training_scripts/coder1.5B/qwen2.5_lora_sft_Glaive-Code-Assistant_finetune.yaml
llamafactory-cli train training_scripts/coder1.5B/qwen2.5_lora_sft_Evol_finetune.yaml
llamafactory-cli train training_scripts/coder1.5B/qwen2.5_lora_sft_Code_ShareGPT_finetune.yaml
llamafactory-cli train training_scripts/coder1.5B/qwen2.5_lora_sft_Python_Codes_finetune.yaml
llamafactory-cli train training_scripts/coder1.5B/qwen2.5_lora_sft_Rosseta_Code_finetune.yaml
llamafactory-cli train training_scripts/coder1.5B/qwen2.5_lora_sft_LLaMA_Python_Code_finetune.yaml
llamafactory-cli train training_scripts/coder1.5B/qwen2.5_lora_sft_Code_Alpaca_finetune.yaml
