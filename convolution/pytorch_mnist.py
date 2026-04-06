import torch
from torchvision.datasets import MNIST
import time
import numpy as np
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard import SummaryWriter


class ConvolutionalModel(nn.Module):
    def __init__(self, in_channels, conv1_width, conv2_width, fc1_width, class_count):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, conv1_width, kernel_size=5, stride=1, padding=2, bias=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(conv1_width, conv2_width, kernel_size=5, stride=1, padding=2, bias=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(conv2_width * 7 * 7, fc1_width, bias=True)
        self.fc_logits = nn.Linear(fc1_width, class_count, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.Linear) and m is not self.fc_logits:
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.constant_(m.bias, 0.)
        self.fc_logits.reset_parameters()

    def forward(self, x):
        h = self.conv1(x)
        h = self.pool1(h)
        h = self.relu(h)

        h = self.conv2(h)
        h = self.pool2(h)
        h = self.relu(h)

        h = h.view(h.shape[0], -1)
        h = self.fc1(h)
        h = self.relu(h)
        logits = self.fc_logits(h)
        return logits


def evaluate(model, loader, criterion, device, name="Validation"):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            total_loss += loss.item() * batch_x.size(0)
            total_correct += (logits.argmax(dim=1) == batch_y).sum().item()
            total_examples += batch_x.size(0)

    avg_loss = total_loss / total_examples
    acc = 100.0 * total_correct / total_examples

    print(f"{name} accuracy = {acc:.2f}")
    print(f"{name} avg loss = {avg_loss:.4f}\n")

    return avg_loss, acc


def save_conv1_filters(writer, model, epoch):
    with torch.no_grad():
        weights = model.conv1.weight.detach().cpu().clone()

        w_min = weights.min()
        w_max = weights.max()
        if w_max > w_min:
            weights = (weights - w_min) / (w_max - w_min)
        else:
            weights = torch.zeros_like(weights)

        grid = make_grid(weights, nrow=8, padding=2, normalize=False)
        writer.add_image("Conv1 Filters", grid, epoch)


DATA_DIR = Path(__file__).parent / 'datasets' / 'MNIST'
max_epochs = 8
batch_size = 50
lr_policy = {1: {'lr': 1e-1}, 3: {'lr': 1e-2}, 5: {'lr': 1e-3}, 7: {'lr': 1e-4}}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

np.random.seed(int(time.time() * 1e6) % 2**31)
torch.manual_seed(int(time.time() * 1e6) % 2**31)

ds_train = MNIST(DATA_DIR, train=True, download=True)
ds_test = MNIST(DATA_DIR, train=False, download=True)

train_x = ds_train.data.reshape([-1, 1, 28, 28]).numpy().astype(np.float32) / 255.0
train_y = ds_train.targets.numpy()
train_x, valid_x = train_x[:55000], train_x[55000:]
train_y, valid_y = train_y[:55000], train_y[55000:]

test_x = ds_test.data.reshape([-1, 1, 28, 28]).numpy().astype(np.float32) / 255.0
test_y = ds_test.targets.numpy()

train_mean = train_x.mean()
train_x, valid_x, test_x = (x - train_mean for x in (train_x, valid_x, test_x))

train_x, valid_x, test_x = (torch.from_numpy(x).float() for x in (train_x, valid_x, test_x))
train_y, valid_y, test_y = (torch.from_numpy(y).long() for y in (train_y, valid_y, test_y))

train_ds = TensorDataset(train_x, train_y)
valid_ds = TensorDataset(valid_x, valid_y)
test_ds = TensorDataset(test_x, test_y)

train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=False)
test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

model = ConvolutionalModel(in_channels=1, conv1_width=16, conv2_width=32, fc1_width=512, class_count=10
).to(device)

criterion = nn.CrossEntropyLoss()

for i, weight_decay in enumerate([1e-1, 1e-2, 1e-3]):
    print(f"Training with weight decay = {weight_decay:.1e}")
    optimizer = optim.SGD(model.parameters(), lr=lr_policy[1]['lr'], weight_decay=weight_decay
    )
    SAVE_DIR = Path(__file__).parent / f"out_pytorch_{i}"
    SAVE_DIR.mkdir(exist_ok=True)

    writer = SummaryWriter(log_dir=SAVE_DIR)

    train_losses = []
    valid_losses = []

    for epoch in range(1, max_epochs + 1):
        if epoch in lr_policy:
            lr = lr_policy[epoch]['lr']
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        model.train()
        running_loss = 0.0
        running_correct = 0
        total_examples = 0

        for step, (batch_x, batch_y) in enumerate(train_dl):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * batch_x.size(0)
            running_correct += (logits.argmax(dim=1) == batch_y).sum().item()
            total_examples += batch_x.size(0)

            if step % 10 == 0:
                print(
                    f"epoch {epoch}, step {step * batch_size}/{len(train_ds)}, "
                    f"batch loss = {loss.item():.4f}, lr = {optimizer.param_groups[0]['lr']:.1e}"
                )
        train_loss = running_loss / total_examples
        train_acc = 100.0 * running_correct / total_examples
        train_losses.append(train_loss)

        print(f"Train accuracy = {train_acc:.2f}") 
        print(f"Train avg loss = {train_loss:.4f}\n")

        val_loss, val_acc = evaluate(model, valid_dl, criterion, device, name="Validation")
        valid_losses.append(val_loss)

        save_conv1_filters(writer, model, epoch)
        
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Validation", val_acc, epoch)
        writer.add_scalar("Learning Rate", optimizer.param_groups[0]['lr'], epoch)

    writer.close()
    test_loss, test_acc = evaluate(model, test_dl, criterion, device, name="Test")

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, marker='o', label='Train loss')
    plt.plot(range(1, len(valid_losses) + 1), valid_losses, marker='s', label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss through epochs')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(SAVE_DIR / "loss_curve.png", dpi=150)
    plt.show()