import torchvision.datasets as datasets
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

import torch

image_transforms = {
    # Train uses data augmentation
    'train':
    transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),  # Image net standards
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])  # Imagenet standards
    ]),
    # Validation does not use augmentation
    'valid':
    transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Datasets from folders
data = {
    'train':
    datasets.ImageFolder(root="/content/drive/My Drive/IMAGE_RECOGNITION/TRAIN", transform=image_transforms['train']),
    'valid':
    datasets.ImageFolder(root="/content/drive/My Drive/IMAGE_RECOGNITION/TEST", transform=image_transforms['valid']),
    'test':
    datasets.ImageFolder(root="/content/drive/My Drive/IMAGE_RECOGNITION/VAL", transform=image_transforms['valid']),
}

# Dataloader iterators, make sure to shuffle
dataloaders = {
    'train': DataLoader(data['train'], batch_size=100, shuffle=True),
    'val': DataLoader(data['valid'], batch_size=10, shuffle=True),
    'test': DataLoader(data['test'], batch_size=1, shuffle=True)
}