import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class CNNModel_FULL(nn.Module):
    def __init__(self, n_labels,input_size=(3, 256, 256)):
        super(CNNModel_FULL, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 1024, kernel_size=3), 
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(1024, 512, kernel_size=5), 
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(512, 256, kernel_size=3),  # 第三个卷积层
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(256, 128, kernel_size=3),  # 第四个卷积层
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.flatten = nn.Flatten()  # 展平层
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)  # 1张虚拟输入图片
            flattened_size = self.conv_layers(dummy_input).view(1, -1).size(1)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_size, 256),  # 全连接层 1
            nn.ELU(),
            nn.Linear(256, 128),  # 全连接层 2
            nn.ELU(),
            nn.Linear(128, 64),  # 全连接层 3
            nn.ELU(),
            nn.Linear(64, 32),  # 全连接层 4
            nn.ELU(),
            nn.Linear(32, n_labels),  # 输出层
            nn.Softmax(dim=1)  # 添加 softmax
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x

class CNNModel_Middle(nn.Module):
    def __init__(self, n_labels,input_size=(3, 256, 256)):
        super(CNNModel_Middle, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 1024, kernel_size=3), 
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(1024, 512, kernel_size=5), 
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(512, 256, kernel_size=3),  # 第三个卷积层
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(256, 128, kernel_size=3),  # 第四个卷积层
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.flatten = nn.Flatten()  # 展平层
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_size)  # 1张虚拟输入图片
            flattened_size = self.conv_layers(dummy_input).view(1, -1).size(1)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(flattened_size, 256),  # 全连接层 1
            nn.ELU(),
            nn.Linear(256, 64),  # 全连接层 1
            nn.ELU(),
            nn.Linear(64, n_labels),
            nn.Softmax(dim=1)  # 添加 softmax
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x    

if __name__ == "__main__":
    n_labels = 15
    model = CNNModel_FULL(n_labels)
    # 检查模型结构
    print('CNN_FULL model is ')
    print(model)
    print('===================Finish================\n')
    
    model = CNNModel_Middle(n_labels)
    print('CNN_Middle model is ')
    print(model)
    print('===================Finish================\n')