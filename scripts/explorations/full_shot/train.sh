export CUDA_VISIBLE_DEVICES=0,1,2,3
cd ./prepare

llamafactory-cli train training_scripts/ablations/full_shot/qwen2.5_lora_sft_arcc_fullshot.yaml
