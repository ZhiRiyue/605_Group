import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from model import CNNModel_FULL
from datasets import train_dataset

# def load_latest_checkpoint(model, optimizer, checkpoints_dir):
#     """
#     从指定的 checkpoints 目录中检索最新模型文件并加载参数。
#     """
#     if not os.path.exists(checkpoints_dir):
#         print(f"Checkpoints directory '{checkpoints_dir}' does not exist.")
#         return model, optimizer, 0

#     checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith(".pth")]
#     if not checkpoint_files:
#         print(f"No checkpoint files found in '{checkpoints_dir}'.")
#         return model, optimizer, 0

#     checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)), reverse=True)
#     latest_checkpoint = checkpoint_files[0]
#     latest_checkpoint_path = os.path.join(checkpoints_dir, latest_checkpoint)
#     print(f"Loading model from checkpoint: {latest_checkpoint_path}")
    
#     # 加载检查点
#     checkpoint = torch.load(latest_checkpoint_path)
#     model.load_state_dict(checkpoint["model_state_dict"])
#     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
#     epoch = checkpoint["epoch"]
#     print(f"Checkpoint loaded: {latest_checkpoint}, Epoch: {epoch}")
    
#     return model, optimizer, epoch
# def load_latest_checkpoint(model, optimizer, checkpoints_dir):
#     """
#     从指定的 checkpoints 目录中检索最新模型文件并加载参数。
#     """
#     if not os.path.exists(checkpoints_dir):
#         print(f"Checkpoints directory '{checkpoints_dir}' does not exist.")
#         return model, optimizer, 0

#     checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith(".pth")]
#     if not checkpoint_files:
#         print(f"No checkpoint files found in '{checkpoints_dir}'.")
#         return model, optimizer, 0

#     # 找到最新的 checkpoint 文件
#     checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)), reverse=True)
#     latest_checkpoint = checkpoint_files[0]
#     latest_checkpoint_path = os.path.join(checkpoints_dir, latest_checkpoint)
#     print(f"Loading model from checkpoint: {latest_checkpoint_path}")
    
#     # 加载检查点文件
#     checkpoint = torch.load(latest_checkpoint_path)
    
#     # 加载模型参数（直接加载 state_dict）
#     if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
#         model.load_state_dict(checkpoint["model_state_dict"])
#         optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
#         epoch = checkpoint["epoch"]
#         print(f"Checkpoint loaded: {latest_checkpoint}, Epoch: {epoch}")
#     else:
#         # 如果检查点文件直接是 state_dict
#         model.load_state_dict(checkpoint)
#         epoch = 0  # 如果没有 epoch 信息，则默认从 0 开始
#         print(f"Checkpoint loaded: {latest_checkpoint}, Starting from epoch: {epoch}")

#     return model, optimizer, epoch


# def load_latest_checkpoint(model, optimizer, checkpoints_dir):
#     """
#     从指定的 checkpoints 目录中检索最新模型文件并加载参数。
#     """
#     if not os.path.exists(checkpoints_dir):
#         print(f"Checkpoints directory '{checkpoints_dir}' does not exist.")
#         return model, optimizer, 0

#     checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith(".pth")]
#     if not checkpoint_files:
#         print(f"No checkpoint files found in '{checkpoints_dir}'.")
#         return model, optimizer, 0

#     # 找到最新的 checkpoint 文件
#     checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)), reverse=True)
#     latest_checkpoint = checkpoint_files[0]
#     latest_checkpoint_path = os.path.join(checkpoints_dir, latest_checkpoint)
#     print(f"Loading model from checkpoint: {latest_checkpoint_path}")
    
#     # 加载检查点文件
#     checkpoint = torch.load(latest_checkpoint_path)
    
#     # 移除 `module.` 前缀
#     new_state_dict = {}
#     for k, v in checkpoint.items():
#         new_key = k.replace("module.", "")  # 移除 `module.`
#         new_state_dict[new_key] = v

#     # 加载新的 state_dict
#     model.load_state_dict(new_state_dict, strict=False)
#     epoch = 0  # 如果没有 epoch 信息，则默认从 0 开始
#     print(f"Checkpoint loaded: {latest_checkpoint}, Starting from epoch: {epoch}")

#     return model, optimizer, epoch


def load_latest_checkpoint(model, optimizer, checkpoints_dir):
    """
    从指定的 checkpoints 目录中检索最新模型文件并加载参数。
    """
    if not os.path.exists(checkpoints_dir):
        print(f"Checkpoints directory '{checkpoints_dir}' does not exist.")
        return model, optimizer, 0

    checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith(".pth")]
    if not checkpoint_files:
        print(f"No checkpoint files found in '{checkpoints_dir}'.")
        return model, optimizer, 0

    # 找到最新的 checkpoint 文件
    checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)), reverse=True)
    latest_checkpoint = checkpoint_files[0]
    latest_checkpoint_path = os.path.join(checkpoints_dir, latest_checkpoint)
    print(f"Loading model from checkpoint: {latest_checkpoint_path}")
    
    # 加载检查点文件
    checkpoint = torch.load(latest_checkpoint_path)
    
    # 移除 `module.` 前缀（如果存在）
    if "model_state_dict" in checkpoint:
        new_state_dict = {}
        for k, v in checkpoint["model_state_dict"].items():
            new_key = k.replace("module.", "")  # 移除 `module.` 前缀
            new_state_dict[new_key] = v
        model.load_state_dict(new_state_dict, strict=False)
    else:
        print("model_state_dict not found in checkpoint, loading entire checkpoint as state_dict.")
        model.load_state_dict(checkpoint, strict=False)

    # 加载 epoch 信息
    if "epoch" in checkpoint:
        epoch = checkpoint["epoch"]
        print(f"Starting from epoch: {epoch}")
    else:
        epoch = 0
        print("Epoch information not found in checkpoint. Starting from epoch 0.")

    # 加载优化器状态
    if "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        print("Optimizer state not found in checkpoint. Initializing optimizer from scratch.")

    return model, optimizer, epoch



def save_checkpoint(model, optimizer, epoch, checkpoint_path):
    """
    保存模型、优化器状态和当前epoch。
    """
    state = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch
    }
    torch.save(state, checkpoint_path)
    print(f"Checkpoint saved to: {checkpoint_path}")

    
    
    
    
# if __name__ == "__main__":
#     # 数据加载器
#     parser = argparse.ArgumentParser(description="Train parallel script.")
#     parser.add_argument("--num_epochs", type=int, required=True, help="Number of epochs")
#     parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
    
#     args = parser.parse_args()
#     num_epochs = args.num_epochs
#     batch_size = args.batch_size
#     print(f"Number of epochs: {num_epochs}")
#     print(f"Batch size: {batch_size}")

#     dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
#     model = CNNModel_FULL(n_labels=15)
    
#     # 检索并加载最新 checkpoint
#     optimizer = optim.Adam(model.parameters(), lr=1e-4)
#     start_epoch = 0
#     model, optimizer, start_epoch = load_latest_checkpoint(model, optimizer, "checkpoints")

#     # 设备设置
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)

#     # 使用 DataParallel 包裹模型
#     if torch.cuda.device_count() > 1:
#         print(f"Using {torch.cuda.device_count()} GPUs for training.")
#         model = nn.DataParallel(model)
#     else:
#         print(f"Using 1 GPU for training.")

#     # 学习率调度器
#     scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, num_epochs // 10), gamma=0.9)

#     # 开始训练
#     model.train()
#     for epoch in range(start_epoch, num_epochs):
#         running_loss = 0.0
#         for batch_idx, (images, labels) in enumerate(dataloader, 1):  # 从 1 开始计数
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             loss = nn.CrossEntropyLoss()(outputs, labels)

#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             # 每 100 个 batch 打印一次损失
#             if batch_idx % 100 == 0:
#                 current_lr = optimizer.param_groups[0]['lr']
#                 print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}, Learning Rate: {current_lr:.6f}")
#             running_loss += loss.item()

#         # 每个 epoch 的平均损失
#         scheduler.step()
#         epoch_loss = running_loss / len(dataloader)
#         print(f"Epoch [{epoch+1}/{num_epochs}] completed, Average Loss: {epoch_loss:.4f}")

#         # 保存模型检查点
#         if (epoch + 1) % max(1, num_epochs // 10) == 0:
#             checkpoint_path = os.path.join("checkpoints", f"model_epoch_{epoch+1}.pth")
#             save_checkpoint(model, optimizer, epoch+1, checkpoint_path)

#     # 保存最终模型
#     final_checkpoint_path = os.path.join("checkpoints", "model_Final.pth")
#     save_checkpoint(model, optimizer, num_epochs, final_checkpoint_path)
#     print(f"Finish training, Checkpoint saved: {final_checkpoint_path}")
if __name__ == "__main__":
    # 数据加载器
    parser = argparse.ArgumentParser(description="Train parallel script.")
    parser.add_argument("--num_epochs", type=int, required=True, help="Number of epochs to train after checkpoint")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
    
    args = parser.parse_args()
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    print(f"Number of additional epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")

    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    model = CNNModel_FULL(n_labels=15)
    
    # 检索并加载最新 checkpoint
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    start_epoch = 0
    model, optimizer, start_epoch = load_latest_checkpoint(model, optimizer, "checkpoints")

    # # 设备设置
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = model.to(device)

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 确保优化器的参数也在同一设备
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


    
    
    
    
    # 使用 DataParallel 包裹模型
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = nn.DataParallel(model)
    else:
        print(f"Using 1 GPU for training.")

    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, (start_epoch + num_epochs) // 10), gamma=0.9)

    # 调整结束的 epoch
    end_epoch = start_epoch + num_epochs
    print(f"Starting training from epoch {start_epoch+1} to {end_epoch}")

    # 开始训练
    model.train()
    for epoch in range(start_epoch, end_epoch):
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(dataloader, 1):  # 从 1 开始计数
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 每 100 个 batch 打印一次损失
            if batch_idx % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch+1}/{end_epoch}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}, Learning Rate: {current_lr:.6f}")
            running_loss += loss.item()

        # 每个 epoch 的平均损失
        scheduler.step()
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{end_epoch}] completed, Average Loss: {epoch_loss:.4f}")

        # 保存模型检查点
        if (epoch + 1) % max(1, (end_epoch - start_epoch) // 10) == 0:
            checkpoint_path = os.path.join("checkpoints", f"model_epoch_{epoch+1}.pth")
            save_checkpoint(model, optimizer, epoch+1, checkpoint_path)

    # 保存最终模型
    final_checkpoint_path = os.path.join("checkpoints", "model_Final.pth")
    save_checkpoint(model, optimizer, end_epoch, final_checkpoint_path)
    print(f"Finish training, Checkpoint saved: {final_checkpoint_path}")
