import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from itertools import product
from collections import OrderedDict
from collections import namedtuple


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=10, kernel_size=5)
        self.fc1 = nn.Linear(in_features=10 * 4 * 4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, t):
        t = F.relu(self.conv1(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        t = F.relu(self.conv2(t))
        t = F.max_pool2d(t, kernel_size=2, stride=2)
        t = F.relu(self.fc1(t.reshape(-1, 10 * 4 * 4)))
        t = F.relu(self.fc2(t))
        t = self.out(t)
        return t


class RunBuilder():
    @staticmethod
    def get_runs(params):
        Run = namedtuple('Run', params.keys())
        runs = []
        for i in product(*params.values()):
            runs.append(Run(*i))
        return runs


class RunManager():
    def __init__(self):
        self.tensorboard = None
        self.epoch = 0
        self.epoch_count = 0

    def beginEpoch(self, run):
        images, labels = next(iter(train_loader))
        grid = torchvision.utils.make_grid(images)
        comment = f'{run}'
        self.tensorboard = SummaryWriter(comment=comment)
        self.tensorboard.add_image('images', grid)
        self.tensorboard.add_graph(network, images)
        self.epoch = 0

    def endEpoch(self):
        self.tensorboard.add_scalar('Number Correct', self.epoch_count, self.epoch)
        self.tensorboard.add_scalar('Accuracy', self.epoch_count / len(train_set), self.epoch)
        print('Epoch {}: {}'.format(self.epoch, self.epoch_count / len(train_set)))
        self.epoch += 1
        self.epoch_count = 0
    def setCount(self, count):
        self.epoch_count += count
    def endTraining(self):
        self.tensorboard.close()
def getCount(preds, lables):
    return preds.argmax(dim=1).eq(lables).sum().item()


if __name__ == '__main__':
    train_set = torchvision.datasets.FashionMNIST(
        root='./data',  # the location on disk where the data is located
        train=True,  # Choosing the train set
        download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )
    paramaters = dict(
        lr=[.001],
        batch_size=[10],
        shuffle=[True],
        num_workers=[0]
    )
    runs = RunBuilder.get_runs(paramaters)
    manager = RunManager()
    for run in runs:
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=run.batch_size,
            shuffle=run.shuffle,
            num_workers=run.num_workers
        )
        network = Network()
        optimizer = optim.Adam(network.parameters(), lr=run.lr)
        manager.beginEpoch(run)
        for epoch in range(1000):
            for batch in train_loader:
                images, lables = batch
                preds = network(images)
                loss = F.cross_entropy(preds, lables)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                manager.setCount(getCount(preds, lables))
            manager.endEpoch()
        torch.save(network.state_dict(), './1stModel')
        manager.endTraining()
