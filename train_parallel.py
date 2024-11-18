import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, utils
from PIL import Image
###########
from model import CNNModel_FULL
from datasets import train_dataset, idx_to_class

if __name__ == "__main__":

    dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)


    model = CNNModel_FULL(n_labels=15)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training")
        model = nn.DataParallel(model)  
    model = model.to(device)


    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)


    num_epochs = 5
    model.train()

    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, (images, labels) in enumerate(dataloader, 1): 
            labels = labels.to(device)
            images = images.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)


            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


            if batch_idx % 5 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

            running_loss += loss.item()
        

        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}] completed, Average Loss: {epoch_loss:.4f}")
