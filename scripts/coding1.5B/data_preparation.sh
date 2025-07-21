export CUDA_VISIBLE_DEVICES=4,5,6,7

cd ./prepare

llamafactory-cli train training_scripts/coder1.5B/qwen2.5_lora_sft_Glaive-Code-Assistant_pretrain.yaml
llamafactory-cli train training_scripts/coder1.5B/qwen2.5_lora_sft_Evol_pretrain.yaml
llamafactory-cli train training_scripts/coder1.5B/qwen2.5_lora_sft_Code_ShareGPT_pretrain.yaml
llamafactory-cli train training_scripts/coder1.5B/qwen2.5_lora_sft_Python_Codes_pretrain.yaml
llamafactory-cli train training_scripts/coder1.5B/qwen2.5_lora_sft_Rosseta_Code_pretrain.yaml
llamafactory-cli train training_scripts/coder1.5B/qwen2.5_lora_sft_LLaMA_Python_Code_pretrain.yaml
llamafactory-cli train training_scripts/coder1.5B/qwen2.5_lora_sft_Code_Alpaca_pretrain.yaml
