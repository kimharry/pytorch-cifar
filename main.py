'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

import matplotlib; matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', '-l', default=0.01, type=float, help='learning rate')
parser.add_argument('-num_epochs', '-n', default=200, type=int, help='number of epochs to train')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--batch_norm', '-b', action='store_true', help='use batch normalization layer')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
if args.batch_norm:
    net = LeNet_BN()
else:
    net = LeNet()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

train_accs = []
train_losses = []
test_accs = []
test_losses = []

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

    train_accs = checkpoint['train_accs']
    train_losses = checkpoint['train_losses']
    test_accs = checkpoint['test_accs']
    test_losses = checkpoint['test_losses']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    
    epoch_total_loss = train_loss / (batch_idx + 1)
    epoch_total_acc = 100. * correct / total
    train_losses.append(epoch_total_loss)
    train_accs.append(epoch_total_acc)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
    

    # Save checkpoint.
    acc = 100.*correct/total
    epoch_total_loss = test_loss / (batch_idx + 1)
    test_losses.append(epoch_total_loss)
    test_accs.append(acc)

    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'train_accs': train_accs,
            'test_accs': test_accs,
            'train_losses': train_losses,
            'test_losses': test_losses,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc


def plot_loss_acc():
    # plot training loss, accuracy, and test loss, accuracy in one figure
    fig, ax1 = plt.subplots(figsize=(10, 8))
    ax2 = ax1.twinx()

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.plot(np.arange(len(train_losses)), train_losses, label='Train Loss', color='blue')
    ax1.plot(np.arange(len(test_losses)), test_losses, label='Test Loss', color='orange')

    ax2.plot(np.arange(len(train_accs)), train_accs, label='Train Accuracy', color='green')
    ax2.plot(np.arange(len(test_accs)), test_accs, label='Test Accuracy', color='red')

    ax2.set_ylabel('Accuracy')

    fig.legend()
    plt.tight_layout()
    plt.grid(linestyle='--', alpha=0.5)
    plt.title('Loss and Accuracy over Epochs')
    plt.xticks(np.arange(len(train_losses)), np.arange(1, len(train_losses) + 1))
    ax1.set_ylim(0, max(max(train_losses), max(test_losses)) + 1)
    ax2.set_ylim(0, 100)
    plt.savefig('loss_acc.png')
    plt.show()

def check_feature_size():
    # out = F.relu(self.conv1(x))
    # out = F.max_pool2d(out, 2)
    # out = F.relu(self.conv2(out))
    # out = F.max_pool2d(out, 2)
    # out = out.view(out.size(0), -1)
    # out = F.relu(self.fc1(out))
    # out = F.relu(self.fc2(out))
    # out = self.fc3(out)
    import pdb

    x, targets = next(iter(trainloader))
    x = x.to(device)
    net = LeNet().to(device)

    pdb.set_trace()
    net.forward(x)

# check_feature_size()

for epoch in range(start_epoch, start_epoch+args.num_epochs):
    train(epoch)
    test(epoch)
    scheduler.step()
    print(f"Learning rate: {scheduler.get_last_lr()[0]}")

plot_loss_acc()

# visualize the first image of the input mini-batch in RGB space. and check the range of each rgb channel
# images, labels = next(iter(trainloader))
# print(f"R: {images[0][0].min()} - {images[0][0].max()}")
# print(f"G: {images[0][1].min()} - {images[0][1].max()}")
# print(f"B: {images[0][2].min()} - {images[0][2].max()}")
# npimg = images.numpy()
# plt.imshow(npimg[0].transpose(1, 2, 0))
# plt.title(f"Label: {classes[labels[0]]}")
# plt.show()