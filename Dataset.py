import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


def unpickle(file):
    with open(file, 'rb') as fo:
        dict_data = pickle.load(fo, encoding='bytes')
    return dict_data


def load_cifar10(data_dir):
    # 加载训练数据
    train_data, train_labels = [], []
    for i in range(1, 6):
        batch = unpickle(f"{data_dir}/data_batch_{i}")
        train_data.append(batch[b'data'])
        train_labels += batch[b'labels']
    train_data = np.concatenate(train_data)

    # 加载测试数据
    test_batch = unpickle(f"{data_dir}/test_batch")
    test_data = test_batch[b'data']
    test_labels = test_batch[b'labels']

    # 转换为 NumPy 数组
    train_data = train_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_data = test_data.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)

    return (train_data, np.array(train_labels)), (test_data, np.array(test_labels))


data_dir = "cifar-10-batches-py"
(train_images, train_labels), (test_images, test_labels) = load_cifar10(data_dir)


class MyDataset(Dataset):
    def __init__(self, images, labels, train_opt=True):
        self.images = images
        self.labels = labels
        self.train_opt = train_opt
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        image = Image.fromarray(image)
        if self.train_opt:
            image = self.transform(image)
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            image = transform(image)
        return image, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.images)


train_dataset = MyDataset(train_images, train_labels)
test_dataset = MyDataset(test_images, test_labels, False)

train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

if __name__ == '__main__':
    batch = unpickle(f"{data_dir}/data_batch_1")
    print(batch[b'data'].shape)
    print(train_images.shape)
