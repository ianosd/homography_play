from pentip_dataset import PenTipDataset
from matplotlib import pyplot as plt
from torch.utils.data import Subset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch
from itertools import cycle
from torchvision.transforms.v2 import ToDtype

data_folder = "data/labeled"
annotations_file = "data/annotations.csv"

dataset = PenTipDataset(annotations_file, data_folder, transform=ToDtype(torch.float32, scale=True))
print(f"Size of dataset: {len(dataset)}")

### Split the full set into 3/4 for training, 1/4 for testing
def training_set_indices():
    i = 0
    increments = [1, 1, 2]
    for increment in cycle(increments):
        if i >= len(dataset):
            break
        yield i
        i += increment

def test_set_indices():
    i = 3
    while i < len(dataset):
        yield i
        i += 4
    
training_set = Subset(dataset, list(training_set_indices()))
test_set = Subset(dataset, list(test_set_indices()))

### Define the model

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=5, padding="same"),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, 5, padding="same"),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 1, 21, padding=10, stride=5),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.stack(x)

model = Net().to("cpu")
print(model)

image, output = test_set[5]
y = model(image)

plt.subplot(131)
plt.imshow(image.permute((1, 2, 0)))
plt.subplot(132)
plt.imshow(output)
plt.subplot(133)
plt.imshow(y.detach().squeeze(0))
plt.show()

# best parameters so far: lr 0.1, bs 10, ep (saturates at 20)
learning_rate = 0.1 
batch_size = 10
epochs = 20

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    num_ok = 0
    for (X, y) in dataloader:
        pred = model(X).squeeze() # output is batch_size x 1 x 5 x 5
        loss = loss_fn(pred, y)
        predictions = torch.argmax(torch.reshape(pred, (pred.shape[0], pred.shape[1]*pred.shape[2])), 1)
        truth = torch.argmax(torch.reshape(y, (y.shape[0], y.shape[1]*y.shape[2])), 1)

        num_ok += sum(predictions == truth)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    return loss.item(), num_ok

train_data_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)

class ProgressShower:
    def __init__(self):
        self.signals = {}
        self.ax = []
        self.fig, ax = plt.subplots()
        self.ax.append(ax)
        self.ax.append(ax.twinx())
        ax.set_xlabel("epoch")
        
    def create_signal(self, name, ax_index):
        line, = self.ax[ax_index].plot([], [], label=name);
        self.signals[name] = [[], [], line]
        
    def add_point_to_signal(self, name, x, y):
        self.signals[name][0].append(x)
        self.signals[name][1].append(y)
        self.signals[name][2].set_xdata(self.signals[name][0])
        self.signals[name][2].set_ydata(self.signals[name][1])
        
        for ax in self.ax:
            ax.relim()
            ax.autoscale_view()

        plt.legend()
        plt.draw()
        plt.pause(0.05)
        
progress = ProgressShower()
progress.create_signal("loss", 0)
progress.create_signal("~ train accuracy", 1)

for epoch in range(epochs):
    loss, train_acc = train_loop(train_data_loader, model, loss_fn, optimizer)
    progress.add_point_to_signal("loss", epoch, loss)
    progress.add_point_to_signal("~ train accuracy", epoch, train_acc)

plt.show()

image, output = test_set[5]
y = model(image)

plt.subplot(131)
plt.imshow(image.permute((1, 2, 0)))
plt.subplot(132)
plt.imshow(output)
plt.subplot(133)
plt.imshow(y.detach().squeeze(0))
plt.show()