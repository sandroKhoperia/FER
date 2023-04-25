import os
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

DATASET_FOLDER = f'datasets'
TRAIN_DIRECTORY = os.path.join(DATASET_FOLDER, "train")
TEST_DIRECTORY = os.path.join(DATASET_FOLDER, "test")

def create_dataset():
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.Resize((48, 48)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = ImageFolder(root=TRAIN_DIRECTORY, transform=train_transform)
    val_dataset = ImageFolder(root=TEST_DIRECTORY, transform=val_transform)

    return train_dataset, val_dataset

class DataModel:
    def __init__(self):
        self.LEARNING_RATE = 0.001
        self.BATCH_SIZE = 128
        self.log_interval = 10
        self.EPOCHS = 100
        self.random_seed = 1
        self.train_losses = []
        self.test_losses = []
        self.train_correct = []
        self.test_correct = []
        torch.backends.cudnn.enabled = False
        torch.manual_seed(self.random_seed)
        self.train_dataset, self.val_dataset = create_dataset()
        self.train_loader = DataLoader(self.train_dataset, batch_size= self.BATCH_SIZE, shuffle=True, num_workers=4)
        self.test_loader = DataLoader(self.val_dataset, batch_size= self.BATCH_SIZE, shuffle=False, num_workers=4)
        self.test_counter = len(self.test_loader.dataset)//self.BATCH_SIZE
        self.train_counter = len(self.train_loader.dataset)//self.BATCH_SIZE
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




def main():
    config = DataModel()
    print(config.train_dataset.classes)
    print(len(config.train_dataset))
    print(len(config.val_dataset))
    print(len(config.test_counter))


if __name__ == "__main__":
    main()