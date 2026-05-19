import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

# 放大batch_size
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16,32,3,padding=1)
        self.fc1 = nn.Linear(32*8*8, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self,x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32*8*8)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_losses, val_losses, train_accs, val_accs = [],[],[],[]
epochs = 3  # 只跑3轮
for epoch in range(epochs):
    model.train()
    train_loss=0; correct=0; total=0
    for images,labels in train_loader:
        images,labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _,pred = torch.max(outputs,1)
        correct += (pred==labels).sum().item()
        total += labels.size(0)
    train_losses.append(train_loss/len(train_loader))
    train_acc = 100*correct/total
    train_accs.append(train_acc)

    model.eval()
    val_loss=0; correct=0; total=0
    with torch.no_grad():
        for images,labels in test_loader:
            images,labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs,labels)
            val_loss += loss.item()
            _,pred = torch.max(outputs,1)
            correct += (pred==labels).sum().item()
            total += labels.size(0)
    val_losses.append(val_loss/len(test_loader))
    val_acc = 100*correct/total
    val_accs.append(val_acc)
    print(f"[{epoch+1}] Train Acc:{train_acc:.2f}% | Val Acc:{val_acc:.2f}%")

# 最终测试准确率
model.eval()
correct=0; total=0
with torch.no_grad():
    for images,labels in test_loader:
        images,labels = images.to(device), labels.to(device)
        outputs = model(images)
        _,pred = torch.max(outputs,1)
        correct += (pred==labels).sum().item()
        total += labels.size(0)
test_acc = 100*correct/total
print(f"\nCIFAR‑10 测试准确率: {test_acc:.2f}%")

# 画图
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.subplot(1,2,2)
plt.plot(train_accs, label='Train Accuracy')
plt.plot(val_accs, label='Val Accuracy')
plt.title('Training & Validation Accuracy')
plt.legend()
plt.show()