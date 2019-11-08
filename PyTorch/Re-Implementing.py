import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def loadTrainDataset():
    train_set = torchvision.datasets.FashionMNIST(
        root='./data',
        download=False,
        train=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=10,
        shuffle=True
    )
    return train_loader, len(train_set)


def loadVerifyDataset():
    verify_set = torchvision.datasets.FashionMNIST(
        root='./data',
        download=False,
        train=False,
        transform=transforms.Compose([transforms.ToTensor()])
    )
    verify_loader = torch.utils.data.DataLoader(
        verify_set
    )
    return verify_loader, len(verify_set)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3)
        self.out = nn.Linear(in_features=6 * 26 * 26, out_features=10)

    def forward(self, t):
        t = torch.sigmoid(self.conv1(t))
        t = torch.sigmoid(self.out(t.reshape(-1, 6 * 26 * 26)))
        return t


def trainModel(train_loader, train_set_len):
    maximum_epoch = 20
    network = CNN()
    optimizer = optim.SGD(network.parameters(), lr=0.01)
    for epoch in range(maximum_epoch):
        train_total_count = 0
        for train_batch in train_loader:
            images, labels = train_batch
            preds = network(images)
            loss = F.cross_entropy(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_total_count += preds.argmax(dim=1).eq(labels).sum().item()
        print('---- Epoch {} ----'.format(epoch))
        print('Train: {}'.format(train_total_count / train_set_len))

    torch.save(network.state_dict(), './2ndModel')


if __name__ == '__main__':
    train_loader, train_set_len = loadTrainDataset()
    verify_loader, verify_set_len = loadVerifyDataset()
    trainModel(train_loader, train_set_len)
