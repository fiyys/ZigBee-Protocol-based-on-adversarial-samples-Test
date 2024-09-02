import os
import shutil

def move_images_to_category_folder(base_dir):
    # 遍历train文件夹下的每个类别文件夹
    for category in os.listdir(base_dir):
        category_path = os.path.join(base_dir, category)
        # 确保是文件夹
        if os.path.isdir(category_path):
            print(f"Processing category: {category}")
            # 遍历类别文件夹下的子文件夹
            for subfolder in os.listdir(category_path):
                subfolder_path = os.path.join(category_path, subfolder)
                # 确保是文件夹
                if os.path.isdir(subfolder_path):
                    print(f"  Processing subfolder: {subfolder}")
                    # 遍历子文件夹下的所有文件
                    for file_name in os.listdir(subfolder_path):
                        # 只处理png文件
                        if file_name.endswith(".png"):
                            source_file = os.path.join(subfolder_path, file_name)
                            destination_file = os.path.join(category_path, file_name)
                            # 移动文件
                            shutil.move(source_file, destination_file)
                            print(f"    Moved {file_name} to {category}")
                    # 删除子文件夹
                    os.rmdir(subfolder_path)
                    print(f"  Deleted subfolder: {subfolder}")

base_dir = "test"  # 请根据实际情况修改路径
move_images_to_category_folder(base_dir)
print("All operations completed.")
