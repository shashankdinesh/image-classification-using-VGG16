from torchvision import models

model = models.vgg16(pretrained=True)

for param in model.parameters():
    param.requires_grad = False
print(model)

import torch.nn as nn
model.classifier[6] = nn.Sequential(
                      nn.Linear(4096, 256),
                      nn.ReLU(),
                      nn.Dropout(0.4),
                      nn.Linear(256, 2),
                      nn.LogSoftmax(dim=1))

total_params = sum(p.numel() for p in model.parameters())
print('{} :, total parameters.'.format(total_params))
total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
print('{} :, training parameters.' .format(total_trainable_params))