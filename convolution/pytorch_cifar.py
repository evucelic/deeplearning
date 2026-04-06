from pathlib import Path
import numpy as np
import torch
from torchvision.datasets import CIFAR10
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import math
import skimage as ski
import skimage.io
import os


def shuffle_data(data_x, data_y):
    indices = np.arange(data_x.shape[0])
    np.random.shuffle(indices)
    shuffled_data_x = np.ascontiguousarray(data_x[indices])
    shuffled_data_y = np.ascontiguousarray(data_y[indices])
    return shuffled_data_x, shuffled_data_y


def evaluate(model, loader, criterion, num_classes, device):
    model.eval()

    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            preds = logits.argmax(dim=1)

            total_loss += loss.item() * batch_x.size(0)
            total_examples += batch_x.size(0)

            for t, p in zip(batch_y.view(-1), preds.view(-1)):
                confusion[t.long(), p.long()] += 1

    avg_loss = total_loss / total_examples
    accuracy = confusion.diag().sum().item() / confusion.sum().item()

    precision = confusion.diag().float() / confusion.sum(dim=0).clamp(min=1).float()
    recall = confusion.diag().float() / confusion.sum(dim=1).clamp(min=1).float()

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "confusion": confusion,
        "precision": precision,
        "recall": recall,
    }


def plot_confusion_matrix(confusion, class_names=None):
    cm = confusion.cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax)

    num_classes = cm.shape[0]
    if class_names is None:
        class_names = [str(i) for i in range(num_classes)]

    ax.set(
        xticks=np.arange(num_classes),
        yticks=np.arange(num_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel='Predicted class',
        ylabel='True class',
        title='Confusion matrix'
    )

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = cm.max() / 2.0 if cm.max() > 0 else 0.0
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > threshold else "black"
            )

    fig.tight_layout()
    return fig


def plot_training_progress(save_dir, data):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 8))

    linewidth = 2
    legend_size = 10
    train_color = 'm'
    val_color = 'c'

    num_points = len(data['train_loss'])
    x_data = np.linspace(1, num_points, num_points)

    ax1.set_title('Cross-entropy loss')
    ax1.plot(x_data, data['train_loss'], marker='o', color=train_color,
             linewidth=linewidth, linestyle='-', label='train')
    ax1.plot(x_data, data['valid_loss'], marker='o', color=val_color,
             linewidth=linewidth, linestyle='-', label='validation')
    ax1.legend(loc='upper right', fontsize=legend_size)

    ax2.set_title('Average class accuracy')
    ax2.plot(x_data, data['train_acc'], marker='o', color=train_color,
             linewidth=linewidth, linestyle='-', label='train')
    ax2.plot(x_data, data['valid_acc'], marker='o', color=val_color,
             linewidth=linewidth, linestyle='-', label='validation')
    ax2.legend(loc='upper left', fontsize=legend_size)

    ax3.set_title('Learning rate')
    ax3.plot(x_data, data['lr'], marker='o', color=train_color,
             linewidth=linewidth, linestyle='-', label='learning_rate')
    ax3.legend(loc='upper left', fontsize=legend_size)

    ax4.axis('off')

    save_path = os.path.join(save_dir, 'training_plot.png')
    print('Plotting in:', save_path)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close(fig)


def draw_conv_filters(epoch, step, weights, save_dir):
    w = weights.copy()
    num_filters = w.shape[0]
    num_channels = w.shape[1]
    k = w.shape[2]
    assert w.shape[3] == w.shape[2]

    w = w.transpose(2, 3, 1, 0)
    w -= w.min()
    if w.max() > 0:
        w /= w.max()

    border = 1
    cols = 8
    rows = math.ceil(num_filters / cols)
    width = cols * k + (cols - 1) * border
    height = rows * k + (rows - 1) * border
    img = np.zeros([height, width, num_channels])

    for i in range(num_filters):
        r = int(i / cols) * (k + border)
        c = int(i % cols) * (k + border)
        img[r:r+k, c:c+k, :] = w[:, :, :, i]

    filename = 'epoch_%02d_step_%06d.png' % (epoch, step)
    img = (img * 255).clip(0, 255).astype(np.uint8)
    ski.io.imsave(os.path.join(save_dir, filename), img)


def show_top_20_worst_misclassified(model, loader, device, class_names, mean, std, save_dir):
    model.eval()
    crit = nn.CrossEntropyLoss(reduction='none')
    bad = []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = crit(logits, y)
            pred = logits.argmax(1)
            prob = torch.softmax(logits, 1)

            for i in (pred != y).nonzero(as_tuple=False).flatten():
                p3 = torch.topk(prob[i], 3)
                bad.append((loss[i].item(), x[i].cpu(), y[i].item(),
                            p3.indices.cpu(), p3.values.cpu()))

    bad.sort(key=lambda t: t[0], reverse=True)

    fig, axes = plt.subplots(4, 5, figsize=(18, 12))
    axes = axes.ravel()

    for ax, (l, img, true_y, topi, topv) in zip(axes, bad[:20]):
        img = img.numpy().transpose(1, 2, 0) * std + mean
        img = np.clip(img, 0, 255).astype(np.uint8)
        ax.imshow(img)
        ax.set_title(
            f"true: {class_names[true_y]}\n"
            f"loss: {l:.2f}\n"
            f"top3: {class_names[topi[0]]}, {class_names[topi[1]]}, {class_names[topi[2]]}",
            fontsize=8
        )
        ax.axis("off")

    for ax in axes[len(bad[:20]):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(Path(SAVE_DIR) / "worst_misclassified.png", dpi=150)
    plt.show()


class ConvolutionModel(nn.Module):
    def __init__(self, in_channels, conv1_width=16, conv2_width=32, fc1_width=256, fc2_width=128, class_count=10):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, conv1_width, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(conv1_width, conv2_width, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(conv2_width * 7 * 7, fc1_width)
        self.fc2 = nn.Linear(fc1_width, fc2_width)
        self.fc_logits = nn.Linear(fc2_width, class_count)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc_logits(x)
        return x


DATA_DIR = Path(__file__).parent / "datasets" / "CIFAR10"
SAVE_DIR = Path(__file__).parent / "out_pytorch_CIFAR10"
SAVE_DIR.mkdir(exist_ok=True)

img_height = 32
img_width = 32
num_channels = 3
num_classes = 10

train_set = CIFAR10(root=DATA_DIR, train=True, download=True)
test_set = CIFAR10(root=DATA_DIR, train=False, download=True)

train_x = train_set.data.astype(np.float32)
train_y = np.array(train_set.targets, dtype=np.int32)

test_x = test_set.data.astype(np.float32)
test_y = np.array(test_set.targets, dtype=np.int32)

valid_size = 5000
train_x, train_y = shuffle_data(train_x, train_y)

valid_x = train_x[:valid_size]
valid_y = train_y[:valid_size]
train_x = train_x[valid_size:]
train_y = train_y[valid_size:]

data_mean = train_x.mean((0, 1, 2))
data_std = train_x.std((0, 1, 2))

train_x = (train_x - data_mean) / data_std
valid_x = (valid_x - data_mean) / data_std
test_x = (test_x - data_mean) / data_std

train_x = train_x.transpose(0, 3, 1, 2)
valid_x = valid_x.transpose(0, 3, 1, 2)
test_x = test_x.transpose(0, 3, 1, 2)

train_x_tensor = torch.from_numpy(train_x).float()
train_y_tensor = torch.from_numpy(train_y).long()
valid_x_tensor = torch.from_numpy(valid_x).float()
valid_y_tensor = torch.from_numpy(valid_y).long()
test_x_tensor = torch.from_numpy(test_x).float()
test_y_tensor = torch.from_numpy(test_y).long()

dataset_train = TensorDataset(train_x_tensor, train_y_tensor)
dataset_valid = TensorDataset(valid_x_tensor, valid_y_tensor)
dataset_test = TensorDataset(test_x_tensor, test_y_tensor)

train_loader = DataLoader(dataset_train, batch_size=50, shuffle=True)
valid_loader = DataLoader(dataset_valid, batch_size=50, shuffle=False)
test_loader = DataLoader(dataset_test, batch_size=50, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ConvolutionModel(
    num_channels,
    conv1_width=16,
    conv2_width=32,
    fc1_width=256,
    fc2_width=128,
    class_count=num_classes
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-2)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
writer = SummaryWriter(log_dir=str(SAVE_DIR))

plot_data = {
    'train_loss': [],
    'valid_loss': [],
    'train_acc': [],
    'valid_acc': [],
    'lr': []
}

class_names = train_set.classes

draw_conv_filters(0, 0, model.conv1.weight.detach().cpu().numpy(), SAVE_DIR)

num_epochs = 50
for epoch in range(1, num_epochs + 1):
    model.train()

    for step, (batch_x, batch_y) in enumerate(train_loader):
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Step {step}/{len(train_loader)}, Loss: {loss.item():.4f}")

        if step % 200 == 0:
            draw_conv_filters(epoch, step, model.conv1.weight.detach().cpu().numpy(), SAVE_DIR)

    train_metrics = evaluate(model, train_loader, criterion, num_classes, device)
    valid_metrics = evaluate(model, valid_loader, criterion, num_classes, device)

    plot_data['train_loss'].append(train_metrics["loss"])
    plot_data['valid_loss'].append(valid_metrics["loss"])
    plot_data['train_acc'].append(train_metrics["accuracy"])
    plot_data['valid_acc'].append(valid_metrics["accuracy"])
    plot_data['lr'].append(optimizer.param_groups[0]["lr"])

    writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
    writer.add_scalar("Loss/valid", valid_metrics["loss"], epoch)
    writer.add_scalar("Accuracy/train", train_metrics["accuracy"], epoch)
    writer.add_scalar("Accuracy/valid", valid_metrics["accuracy"], epoch)
    writer.add_scalar("LR", optimizer.param_groups[0]["lr"], epoch)

    train_cm_fig = plot_confusion_matrix(train_metrics["confusion"], class_names)
    valid_cm_fig = plot_confusion_matrix(valid_metrics["confusion"], class_names)

    writer.add_figure("ConfusionMatrix/train", train_cm_fig, epoch)
    writer.add_figure("ConfusionMatrix/valid", valid_cm_fig, epoch)

    plt.close(train_cm_fig)
    plt.close(valid_cm_fig)

    print(f"\nEpoch {epoch}")
    print(f"Train loss: {train_metrics['loss']:.4f}, acc: {train_metrics['accuracy']:.4f}")
    print(f"Valid loss: {valid_metrics['loss']:.4f}, acc: {valid_metrics['accuracy']:.4f}")
    print("Valid confusion matrix:")
    print(valid_metrics["confusion"].cpu().numpy())
    print("Valid precision per class:")
    print(valid_metrics["precision"].cpu().numpy())
    print("Valid recall per class:")
    print(valid_metrics["recall"].cpu().numpy())
    print()

    scheduler.step()

plot_training_progress(SAVE_DIR, plot_data)

test_metrics = evaluate(model, test_loader, criterion, num_classes, device)
print(f"Test loss: {test_metrics['loss']:.4f}, acc: {test_metrics['accuracy']:.4f}")

show_top_20_worst_misclassified(
    model=model,
    loader=test_loader,
    device=device,
    class_names=class_names,
    mean=data_mean,
    std=data_std,
    save_dir=SAVE_DIR,
)

writer.close()