# import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader
# import argparse
# from model import CNNModel_FULL
# # from datasets import train_dataset

# def load_checkpoint(model, checkpoint_path):
#     """
#     从指定路径加载检查点。
#     """
#     if not os.path.exists(checkpoint_path):
#         print(f"Checkpoint file '{checkpoint_path}' does not exist.")
#         return model, None, None

#     print(f"Loading model from checkpoint: {checkpoint_path}")
#     checkpoint = torch.load(checkpoint_path)
#     model.load_state_dict(checkpoint['model_state_dict'])
#     optimizer_state = checkpoint.get('optimizer_state_dict', None)
#     epoch = checkpoint.get('epoch', None)
#     return model, optimizer_state, epoch

# model = CNNModel_FULL(n_labels=15)

# load_checkpoint(model,"model_Final.pth")


import os
import torch
import torch.nn as nn
from model import CNNModel_FULL

def load_checkpoint(model, checkpoint_path, map_location=None):
    """
    从指定路径加载检查点，同时支持设备映射。
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file '{checkpoint_path}' does not exist.")
        return model, None, None

    # 自动选择设备（如果未指定）
    if map_location is None:
        map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading model from checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    
    # 如果检查点包含`state_dict`，提取
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # 检查是否需要去掉`module.`前缀
    if any(key.startswith("module.") for key in state_dict.keys()):
        print("Removing 'module.' prefix from state_dict keys.")
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}

    model.load_state_dict(state_dict)
    print("Model state_dict loaded successfully.")
    
    # # 获取其他信息（可选）
    # optimizer_state = checkpoint.get("optimizer_state_dict", None)
    # epoch = checkpoint.get("epoch", None)
    
    return model

def load_epoch_checkpoint(checkpoint_path):
    """
    从指定路径加载检查点的训练轮次信息。
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file '{checkpoint_path}' does not exist.")
        return None

    print(f"Loading epoch information from checkpoint: {checkpoint_path}")
    epoch = torch.load(checkpoint_path)

    if not isinstance(epoch, int):
        print("Invalid format: epoch checkpoint must be an integer.")
        return None

    print(f"Epoch information loaded successfully: {epoch}")
    return epoch

def load_optimizer_checkpoint(optimizer, checkpoint_path):
    """
    从指定路径加载检查点的优化器状态。
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file '{checkpoint_path}' does not exist.")
        return optimizer

    print(f"Loading optimizer state from checkpoint: {checkpoint_path}")
    optimizer_state = torch.load(checkpoint_path)

    if not isinstance(optimizer_state, dict):
        print("Invalid format: optimizer checkpoint must be a state dictionary.")
        return optimizer

    optimizer.load_state_dict(optimizer_state)
    print("Optimizer state_dict loaded successfully.")
    return optimizer

def load_scheduler_checkpoint(scheduler, checkpoint_path):
    """
    从指定路径加载检查点的学习率调度器状态。
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint file '{checkpoint_path}' does not exist.")
        return scheduler

    print(f"Loading scheduler state from checkpoint: {checkpoint_path}")
    scheduler_state = torch.load(checkpoint_path)

    if not isinstance(scheduler_state, dict):
        print("Invalid format: scheduler checkpoint must be a state dictionary.")
        return scheduler

    scheduler.load_state_dict(scheduler_state)
    print("Scheduler state_dict loaded successfully.")
    return scheduler

if __name__ == "__main__":
    # 模型定义
    model = CNNModel_FULL(n_labels=15)
    print("Model initiallize!")
    print(model)
    print('==============================================')
    # 加载检查点
    model= load_checkpoint(model, "model_Final.pth")
    print("Model loaded successfully!")
    print(model)
    print('==============================================')