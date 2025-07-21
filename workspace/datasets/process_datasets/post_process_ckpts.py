import os
import shutil


def delete_non_safetensor_files(directory):
    """
    delete all files except .safetensor in designated directory.
    """

    for root, dirs, files in os.walk(directory, topdown=False):
        for file_name in files:
            if not file_name.endswith(".safetensors"):
                file_path = os.path.join(root, file_name)
                os.remove(file_path)
                print(f"Deleted file: {file_path}")

        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            shutil.rmtree(dir_path)
            print(f"Deleted directory: {dir_path}")

    for dir_name in os.listdir(directory):
        dir_path = os.path.join(directory, dir_name)
        if os.path.isdir(dir_path) and not os.listdir(dir_path):
            os.rmdir(dir_path)
            print(f"Deleted empty directory: {dir_path}")


def process_checkpoints(root_directory):
    """
    rename checkpoint-xxxx to xxxx.safetensors and delete original directory
    """

    for dir_name in os.listdir(root_directory):
        dir_path = os.path.join(root_directory, dir_name)

        if os.path.isdir(dir_path) and dir_name.startswith("checkpoint-"):
            try:
                checkpoint_number = dir_name.split("-")[1]

                safetensors_path = os.path.join(dir_path, "adapter_model.safetensors")

                if os.path.exists(safetensors_path):
                    new_safetensors_name = f"{checkpoint_number}.safetensors"
                    new_safetensors_path = os.path.join(root_directory, new_safetensors_name)

                    shutil.move(safetensors_path, new_safetensors_path)
                    print(f"Moved and renamed {safetensors_path} to {new_safetensors_path}")

                shutil.rmtree(dir_path)
                print(f"Deleted directory: {dir_path}")

            except Exception as e:
                print(f"Error processing {dir_path}: {e}")


if __name__ == "__main__":
    dire = ""
    for directory_to_clean in os.listdir(dire):
        directory_to_clean = os.path.join(dire, directory_to_clean)
        for folder in os.listdir(directory_to_clean):
            try:
                delete_non_safetensor_files(os.path.join(directory_to_clean, folder))
            except:
                print(directory_to_clean)
                continue
        process_checkpoints(directory_to_clean)
        delete_non_safetensor_files(directory_to_clean)
