import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from model import CNNModel_FULL
from datasets import train_dataset
from load_checkpoints import load_checkpoint,load_optimizer_checkpoint,load_epoch_checkpoint,load_scheduler_checkpoint

if __name__ == "__main__":
    # 数据加载器
    parser = argparse.ArgumentParser(description="Train parallel script.")
    parser.add_argument("--num_epochs", type=int, required=True, help="Number of epochs")
    parser.add_argument("--total_num_epochs", type=int, required=True, help="Total Number of epochs")
    parser.add_argument("--batch_size", type=int, required=True, help="Batch size")
    
    args = parser.parse_args()
    num_epochs = args.num_epochs
    total_num_epochs = args.total_num_epochs
    batch_size = args.batch_size
    print(f"Number of epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")

    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    model = CNNModel_FULL(n_labels=15)
    print("Model initiallize!")
    print(model)
    print('==============================================')
    # 加载检查点
    model= load_checkpoint(model, "model_Final.pth")
    print("Model loaded successfully!")
    print(model)
    print('==============================================')

    # 设备设置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 使用 DataParallel 包裹模型
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = nn.DataParallel(model)
    else:
        print(f"Using 1 GPU for training.")

    # 训练参数
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    ######### 修改step_size ################################################
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size= 10, gamma=0.9)
    start_epoch = 0
    ######### 增加 ##########################################################
    try:
        optimizer = load_optimizer_checkpoint(optimizer, 'optimizer.pth')
        print('Successfully load optimizer')
    except:
        print('no optimizer.pth')
    try:
        scheduler = load_scheduler_checkpoint(scheduler, 'scheduler.pth')
        print('Successfully load scheduler')
        #new_lr = 2e-5  # 新的学习率
        #for param_group in optimizer.param_groups:
        #    param_group['lr'] = new_lr
        #print(f"Updated learning rate to: {new_lr}")
    except:
        print('no scheduler.pth')
    try:
        start_epoch = load_epoch_checkpoint('epoch.pth')
        print(f'Successfully load start epoch = {start_epoch}')
    except:
        print('no epoch.pth')
    # 开始训练
    model.train()
    ii = 0
    ############################## 修改range(start_epoch, num_epochs) ############################################
    for epoch in range(start_epoch, (start_epoch+num_epochs)):
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(dataloader, 1):  # 从 1 开始计数
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 每 100 个 batch 打印一次损失
            if batch_idx % 100 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch [{epoch+1}/{total_num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}, Learning Rate: {current_lr:.6f}")
            running_loss += loss.item()

        # 每个 epoch 的平均损失
        scheduler.step()
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{total_num_epochs}] completed, Average Loss: {epoch_loss:.4f}")

        # 保存模型检查点
        if (epoch + 1) % max(1, num_epochs // 10) == 0:
            checkpoint_path =f"model_stage_{ii}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
            ii+=1

    # 保存最终模型
    checkpoint_path = os.path.join("model_Final.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Finish training, Checkpoint saved: {checkpoint_path}")
    ##################### 添加 #########################################
    optimizer_path = os.path.join("optimizer.pth")
    torch.save(optimizer.state_dict(), optimizer_path)
    print(f"Optimizer checkpoint saved to: {optimizer_path}")
    
    scheduler_path = os.path.join("scheduler.pth")
    torch.save(scheduler.state_dict(), scheduler_path)
    print(f"Scheduler checkpoint saved to: {scheduler_path}")

    epoch_path = os.path.join("epoch.pth")
    torch.save((start_epoch+num_epochs), epoch_path)
    print(f"Epoch = {(start_epoch+num_epochs)}, checkpoint saved to: {epoch_path}")
