import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

train_set = torchvision.datasets.FashionMNIST(
    root='./data',  # the location on disk where the data is located
    train=True,  # Choosing the train set
    download=True,
    transform=transforms.Compose(transforms.ToTensor())
)
train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=10,
    shuffle=True
)
'''The first sample from the train set'''
sample = next(iter(train_set))
image, label = sample
plt.imshow(image.squeeze(), cmap='gray')
plt.show()


'''Graph the first batch'''
batch = next(iter(train_loader))
images, labels = batch

grid = torchvision.utils.make_grid(images, nrow=10)
plt.figure(figsize=(20, 20))
plt.imshow(grid.permute(1, 2, 0))
# For a colored image... plt.imshow takes image dimension in following form [height width channels] ...
# while pytorch follows [channels height width]...
# so we have to convert (0,1,2) to (1,2,0) form to make it compatible for imshow....
plt.show()


''''''