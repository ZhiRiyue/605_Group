import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import CNNModel_FULL
from datasets import train_dataset

def load_latest_checkpoint(model, checkpoints_dir):
    """
    从指定的 checkpoints 目录中检索最新模型文件并加载参数。
    """
    if not os.path.exists(checkpoints_dir):
        print(f"Checkpoints directory '{checkpoints_dir}' does not exist.")
        return model

    checkpoint_files = [f for f in os.listdir(checkpoints_dir) if f.endswith(".pth")]
    if not checkpoint_files:
        print(f"No checkpoint files found in '{checkpoints_dir}'.")
        return model

    checkpoint_files.sort(key=lambda x: os.path.getmtime(os.path.join(checkpoints_dir, x)), reverse=True)
    latest_checkpoint = checkpoint_files[0]
    latest_checkpoint_path = os.path.join(checkpoints_dir, latest_checkpoint)
    print(f"Loading model from checkpoint: {latest_checkpoint_path}")
    model.load_state_dict(torch.load(latest_checkpoint_path))
    return model

if __name__ == "__main__":
    # 数据加载器
    num_epochs = int(input("Please enter the number of epochs: "))
    batch_size = int(input("Please enter the batch size: "))
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    model = CNNModel_FULL(n_labels=15)
    
    # 检索并加载最新 checkpoint
    print("Try to load model from checkpoints")
    model = load_latest_checkpoint(model, "checkpoints")

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 使用 DataParallel 包裹模型
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = nn.DataParallel(model)

    # 训练参数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=max(1, num_epochs // 10), gamma=0.9)

    # 开始训练
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(dataloader, 1):  # 从 1 开始计数
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 每 5 个 batch 打印一次损失
            if batch_idx % 5 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}, Learning Rate: {current_lr:.6f}")
            running_loss += loss.item()

        # 每个 epoch 的平均损失
        scheduler.step()
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] completed, Average Loss: {epoch_loss:.4f}")

        # 保存模型检查点
        if (epoch + 1) % max(1, num_epochs // 10) == 0:
            checkpoint_path = os.path.join("checkpoints", f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    # 保存最终模型
    checkpoint_path = os.path.join("checkpoints", f"model_Final.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Finish training, Checkpoint saved: {checkpoint_path}")
