import os
import torch
# 指定路径
folder_path = "checkpoints"  

#检查路径是否存在
if os.path.exists(folder_path):
    # 列出所有 .pth 文件
    pth_files = [f for f in os.listdir(folder_path) if f.endswith('.pth')]
    if pth_files:
        print("Available .pth files:")
        for file in pth_files:
            print(file)
    else:
        print("No .pth files found in the directory.")
else:
    print(f"The folder '{folder_path}' does not exist.")



checkpoint = torch.load("checkpoints/model_Final.pth")  
print("Checkpoint keys:", checkpoint.keys())
if "epoch" in checkpoint:
    print("Epoch information:", checkpoint["epoch"])
else:
    print("No epoch information found.")



for file in pth_files:
    file_path = os.path.join(folder_path, file)
    print(f"Checking file: {file}")
    try:
        checkpoint = torch.load(file_path)
        print(f"Keys in {file}:", checkpoint.keys())
    except Exception as e:
        print(f"Failed to load {file}: {e}")

        
        