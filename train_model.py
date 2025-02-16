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

# image, output = test_set[0]
# y = model(image)

# plt.subplot(131)
# plt.imshow(image.permute((1, 2, 0)))
# plt.subplot(132)
# plt.imshow(output)
# plt.subplot(133)
# plt.imshow(y.detach().squeeze(0))
# plt.show()

learning_rate = 1e-1
batch_size = 10
epochs = 100

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch_no, (X, y) in enumerate(dataloader):
        pred = model(X).squeeze() # TODO why does the output have extra dim?
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"After batch {batch_no}/{size} loss: {loss.item():>f}")

train_data_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)

for epoch in range(epochs):
    print(f"Epoch {epoch+1}/epochs")
    train_loop(train_data_loader, model, loss_fn, optimizer)