import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt
from torchvision import datasets
from torchvision.transforms import v2


class DataExtraction: 
    def __init__(self):

        transform = v2.Compose([
            v2.ToImage(),                               # Converts to image tensor
            v2.ToDtype(torch.float32, scale=True),      # Scale pixel values to [0,1]
            v2.Normalize(mean=[0.5], std=[0.5])            # Normalize to [-1,1]
        ])

        self.mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        self.mnist_test = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        self.batch_size = 64
        self.train_dataloader = DataLoader(self.mnist_train, batch_size=self.batch_size)
        self.test_dataloader = DataLoader(self.mnist_test, batch_size=self.batch_size)

    def print_first_batch(self):
        images, labels = next(iter(self.train_dataloader))
        fig, axes = plt.subplots(8, 8, figsize=(8, 8))
        axes = axes.flatten()
        for i in range(self.batch_size):
            img = images[i].squeeze().numpy()
            axes[i].imshow(img, cmap='gray')
            axes[i].axis('off')
        plt.tight_layout()
        plt.show()


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 28x28 → 28x28
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # 28x28 → 14x14
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # 14x14 → 14x14
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # 14x14 → 7x7
        )

        self.fc_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.fc_stack(x)
        return x


def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss_value = loss.item()
            current = (batch + 1) * len(X)
            print(f"loss: {loss_value:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


   


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    data = DataExtraction()
    model = CNN().to(device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    epochs = 15
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(data.train_dataloader, model, loss_fn, optimizer, device)
        test(data.test_dataloader, model, loss_fn, device)
    print("Done!")

    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")

