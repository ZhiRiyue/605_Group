from torch.utils.data import Dataset, DataLoader
from torchvision import datasets,transforms,utils
train_dir = './traindata'
test_dir = './testdata'

# 定义数据变换
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # 调整图像大小为 256x256
    transforms.ToTensor(),          # 将图像转换为 Tensor
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
])

class_to_idx = {
    'Beetle': 0,
    'Butterfly': 1,
    'Cat': 2,
    'Cow': 3,
    'Dog': 4,
    'Elephant': 5,
    'Gorilla': 6,
    'Hippo': 7,
    'Lizard': 8,
    'Monkey': 9,
    'Mouse': 10,
    'Panda': 11,
    'Spider': 12,
    'Tiger': 13,
    'Zebra': 14
    }

idx_to_class = {
    0: 'Beetle',
    1: 'Butterfly',
    2: 'Cat',
    3: 'Cow',
    4: 'Dog',
    5: 'Elephant',
    6: 'Gorilla',
    7: 'Hippo',
    8: 'Lizard',
    9: 'Monkey',
    10: 'Mouse',
    11: 'Panda',
    12: 'Spider',
    13: 'Tiger',
    14: 'Zebra'
    }

# 加载训练集
train_dataset = datasets.ImageFolder(root=train_dir,transform=transform)
test_dataset = datasets.ImageFolder(root=test_dir,transform=transform)

class TargetTransform:
    def __init__(self, class_to_idx, classes):
        self.class_to_idx = class_to_idx
        self.classes = classes

    def __call__(self, target):
        return self.class_to_idx[self.classes[target]]

train_dataset.target_transform = TargetTransform(class_to_idx, train_dataset.classes)
test_dataset.target_transform = TargetTransform(class_to_idx, test_dataset.classes)