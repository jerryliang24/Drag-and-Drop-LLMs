export CUDA_VISIBLE_DEVICES=0,1,2,3
cd  ./prepare

llamafactory-cli train training_scripts/multimodal/qwen2.5vl_lora_MathV360K_pretrain.yaml
llamafactory-cli train training_scripts/multimodal/qwen2.5vl_lora_MathV360K_finetune.yaml
