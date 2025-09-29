#!/bin/bash

# enter the folder containing checkpoints, usually: 
# folder_name:
#  - checkpoint-n
#  - checkpoint-n+1
#  - ...

# cd /path/to/your/checkpoints

# enable globstar to traverse sub directory
shopt -s globstar

# traverse all checkpoint-* folders
for dir in checkpoint-*; do
    # check if it is a directory
    if [[ -d "$dir" ]]; then
        # extract the number part: checkpoint-123 -> 123
        basename=$(basename "$dir")
        number="${basename#checkpoint-}"

        # target file path
        src_file="$dir/adapter_model.safetensors"
        dst_file="$number.safetensors"

        # check if safetensors file exists
        if [[ -f "$src_file" ]]; then
            # 1. delete other files in the directory
            echo "clean dir: $dir"
            find "$dir" -type f ! -name "adapter_model.safetensors" -delete

            # 2. rename and move file to the root folder
            echo "move: $src_file -> $dst_file"
            mv "$src_file" "$dst_file"

            # 3. delete empty directory
            rmdir "$dir"
            echo "deleted dir: $dir"
        else
            echo "⚠️  skip $dir: for lack of adapter_model.safetensors"
        fi
    fi
done

echo "✅ processing complete!"
