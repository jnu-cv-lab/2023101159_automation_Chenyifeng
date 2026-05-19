# ========== 任务1：环境准备 ==========
import torch
import torchvision
import numpy
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim

print("torch版本:", torch.__version__)
print("GPU是否可用:", torch.cuda.is_available())
a = torch.tensor([1,2,3])
print("张量测试正常:", a)

# ========== 任务2：数据集加载 ==========
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# ========== 任务3：搭建CNN模型 ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN().to(device)    

# ========== 任务4：训练模型 ==========
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
epochs = 5
train_loss_list = []
train_acc_list = []

print("========== 开始任务4训练 ==========")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    train_loss_list.append(epoch_loss)
    train_acc_list.append(epoch_acc)
    print(f"Epoch {epoch+1} | Training Loss: {epoch_loss:.4f} | Training Accuracy: {epoch_acc:.2f}%")
print("========== 任务4完成 ==========")

# ===================== 任务5：验证模型 =====================
val_loss_list = []
val_acc_list = []

print("========== 开始任务5验证 ==========")
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct / total

    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
    avg_val_loss = val_loss / len(val_loader)
    avg_val_acc = 100 * correct_val / total_val

    val_loss_list.append(avg_val_loss)
    val_acc_list.append(avg_val_acc)

    print(f"Epoch {epoch+1}")
    print(f"训练 | Loss:{train_loss:.4f}, Acc:{train_acc:.2f}%")
    print(f"验证 | Loss:{avg_val_loss:.4f}, Acc:{avg_val_acc:.2f}%\n")

print("========== 任务5完成 ==========")

# ===================== 任务6：测试模型 =====================
model.eval()
test_loss = 0.0
correct_test = 0
total_test = 0
test_images = []
test_labels_true = []
test_labels_pred = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()

        _, predicted = torch.max(outputs, 1)
        total_test += labels.size(0)
        correct_test += (predicted == labels).sum().item()

        if len(test_images) < 8:
            test_images.extend(images.cpu())
            test_labels_true.extend(labels.cpu().numpy())
            test_labels_pred.extend(predicted.cpu().numpy())

avg_test_loss = test_loss / len(test_loader)
avg_test_acc = 100 * correct_test / total_test
print(f"测试集 Loss: {avg_test_loss:.4f}")
print(f"测试集 Accuracy: {avg_test_acc:.2f}%")

plt.figure(figsize=(12, 6))
for i in range(8):
    plt.subplot(2, 4, i+1)
    plt.imshow(test_images[i].squeeze(), cmap="gray")
    plt.title(f"True:{test_labels_true[i]} Pred:{test_labels_pred[i]}")
    plt.axis("off")
plt.tight_layout()
plt.savefig("test_result.png")  # 保存图片
plt.close()
print("✅ 测试图片已保存：test_result.png")

# ===================== 任务7：绘制训练&验证曲线 =====================
plt.rcParams["font.family"] = "DejaVu Sans"
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(range(1, epochs+1), train_loss_list, label="Training Loss", marker="o")
plt.plot(range(1, epochs+1), val_loss_list, label="Validation Loss", marker="s")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training & Validation Loss Curve")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(range(1, epochs+1), train_acc_list, label="Training Accuracy", marker="o")
plt.plot(range(1, epochs+1), val_acc_list, label="Validation Accuracy", marker="s")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training & Validation Accuracy Curve")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("curve.png")  # 保存图片
plt.close()
print("✅ 训练曲线已保存：curve.png")

print("\n🎉 全部任务 1~7 完成！")