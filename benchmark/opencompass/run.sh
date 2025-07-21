LORA_PATH="/mnt/petrelfs/xuzhaopan/zxl/Drag-and-Drop-LLMs/benchmark/opemcompass/3001.safetensors"
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
OUT_PATH="merged_ckpt"

python ../merge_lora.py --lora_model_path $LORA_PATH --base_model_name $MODEL_PATH --output_dir $OUT_PATH

opencompass --datasets gsm8k_gen_1d7fe4 \
 --hf-type chat \
 --hf-path $OUT_PATH \
 --accelerator vllm

opencompass --datasets math_gen_1ed9c2 \
 --hf-type chat \
 --hf-path $OUT_PATH \
 --accelerator vllm

rm -rf $OUT_PATH
