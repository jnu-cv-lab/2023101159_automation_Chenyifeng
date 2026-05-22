import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ===================== 全局配置 =====================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")
batch_size = 64
epochs = 5  # 可根据需要调整训练轮数

# ===================== 任务1：数据准备与基础模型 =====================
# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载数据集
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 划分训练集和验证集
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 定义CNN模型
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

# 训练函数
def train_model(model, optimizer, criterion, epochs):
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss_avg = train_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss_avg)
        train_accs.append(train_acc)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss_avg = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_losses.append(val_loss_avg)
        val_accs.append(val_acc)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"训练 | Loss: {train_loss_avg:.4f}, Acc: {train_acc:.2f}%")
        print(f"验证 | Loss: {val_loss_avg:.4f}, Acc: {val_acc:.2f}%\n")
    
    # 测试阶段
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            test_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss_avg = test_loss / len(test_loader)
    test_acc = 100 * correct / total
    print(f"测试 | Loss: {test_loss_avg:.4f}, Acc: {test_acc:.2f}%")
    
    return train_losses, val_losses, train_accs, val_accs, test_acc, all_preds, all_labels

# 绘制训练曲线函数
def plot_curves(train_losses, val_losses, train_accs, val_accs, title, save_path):
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs+1), train_losses, label='Training Loss', marker='o')
    plt.plot(range(1, epochs+1), val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{title} - Loss Curve')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs+1), train_accs, label='Training Accuracy', marker='o')
    plt.plot(range(1, epochs+1), val_accs, label='Validation Accuracy', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{title} - Accuracy Curve')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 曲线已保存: {save_path}")

# ===================== 任务1：基础模型训练 =====================
print("="*50)
print("任务1：基础模型训练（Adam优化器，lr=0.001）")
print("="*50)

model_base = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer_base = optim.Adam(model_base.parameters(), lr=0.001)

train_losses_base, val_losses_base, train_accs_base, val_accs_base, test_acc_base, preds_base, labels_base = train_model(
    model_base, optimizer_base, criterion, epochs
)

plot_curves(train_losses_base, val_losses_base, train_accs_base, val_accs_base, 
            "基础模型训练", "task1_base_model_curve.png")

# ===================== 任务2：优化器对比 =====================
print("\n" + "="*50)
print("任务2：优化器对比（SGD vs SGD+Momentum vs Adam）")
print("="*50)

# 定义三种优化器
optimizers = {
    'SGD': optim.SGD(CNN().to(device).parameters(), lr=0.001),
    'SGD+Momentum': optim.SGD(CNN().to(device).parameters(), lr=0.001, momentum=0.9),
    'Adam': optim.Adam(CNN().to(device).parameters(), lr=0.001)
}

results = {}
all_curves = {}

for name, optimizer in optimizers.items():
    print(f"\n正在训练 {name} 优化器...")
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    # 重新初始化优化器（因为上面的优化器绑定了错误的模型参数）
    if name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=0.001)
    elif name == 'SGD+Momentum':
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    train_losses, val_losses, train_accs, val_accs, test_acc, _, _ = train_model(
        model, optimizer, criterion, epochs
    )
    
    results[name] = test_acc
    all_curves[name] = (train_losses, val_losses, train_accs, val_accs)
    plot_curves(train_losses, val_losses, train_accs, val_accs, 
                f"{name} 优化器训练", f"task2_{name}_curve.png")

# 绘制对比曲线
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for name, (train_losses, _, _, _) in all_curves.items():
    plt.plot(range(1, epochs+1), train_losses, label=f'{name} Train Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('不同优化器训练Loss对比')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
for name, (_, _, train_accs, _) in all_curves.items():
    plt.plot(range(1, epochs+1), train_accs, label=f'{name} Train Acc', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('不同优化器训练Accuracy对比')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('task2_optimizers_comparison.png', dpi=300)
plt.close()
print("✅ 优化器对比曲线已保存: task2_optimizers_comparison.png")

# 打印测试准确率对比
print("\n优化器测试准确率对比:")
for name, acc in results.items():
    print(f"{name}: {acc:.2f}%")

# ===================== 任务3：学习率对比 =====================
print("\n" + "="*50)
print("任务3：学习率对比（Adam优化器，lr=0.1, 0.01, 0.001）")
print("="*50)

learning_rates = [0.1, 0.01, 0.001]
lr_results = {}
lr_curves = {}

for lr in learning_rates:
    print(f"\n正在训练学习率 {lr}...")
    model = CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses, val_losses, train_accs, val_accs, test_acc, _, _ = train_model(
        model, optimizer, criterion, epochs
    )
    
    lr_results[lr] = test_acc
    lr_curves[lr] = (train_losses, val_losses, train_accs, val_accs)
    plot_curves(train_losses, val_losses, train_accs, val_accs, 
                f"学习率 {lr} 训练", f"task3_lr_{lr}_curve.png")

# 绘制学习率对比曲线
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
for lr, (train_losses, _, _, _) in lr_curves.items():
    plt.plot(range(1, epochs+1), train_losses, label=f'lr={lr} Train Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('不同学习率训练Loss对比')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
for lr, (_, _, train_accs, _) in lr_curves.items():
    plt.plot(range(1, epochs+1), train_accs, label=f'lr={lr} Train Acc', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('不同学习率训练Accuracy对比')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('task3_learning_rates_comparison.png', dpi=300)
plt.close()
print("✅ 学习率对比曲线已保存: task3_learning_rates_comparison.png")

# 打印测试准确率对比
print("\n学习率测试准确率对比:")
for lr, acc in lr_results.items():
    print(f"lr={lr}: {acc:.2f}%")

# ===================== 任务4：卷积核可视化 =====================
print("\n" + "="*50)
print("任务4：第一层卷积核可视化")
print("="*50)

# 使用任务1训练好的模型
model = model_base
conv1_weights = model.conv1.weight.data.cpu().numpy()

# 显示前16个卷积核（3x3）
plt.figure(figsize=(8, 8))
for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.imshow(conv1_weights[i, 0, :, :], cmap='gray')
    plt.title(f'Kernel {i+1}')
    plt.axis('off')

plt.tight_layout()
plt.savefig('task4_conv1_kernels.png', dpi=300)
plt.close()
print("✅ 第一层卷积核已保存: task4_conv1_kernels.png")
print("说明：训练后的卷积核呈现出不同的边缘、方向和纹理特征，如水平边缘、垂直边缘、斜向边缘等。")
print("卷积核通过反向传播算法不断调整权重，使得损失函数最小化，从而学习到图像的有效特征。")

# ===================== 任务5：Feature map可视化 =====================
print("\n" + "="*50)
print("任务5：第一层卷积输出Feature map可视化")
print("="*50)

# 获取一张测试图片
dataiter = iter(test_loader)
images, labels = next(dataiter)
img = images[0:1].to(device)
true_label = labels[0].item()

# 提取第一层卷积输出
with torch.no_grad():
    conv1_output = model.conv1(img)
    relu_output = model.relu(conv1_output)

# 显示原图
plt.figure(figsize=(12, 6))
plt.subplot(2, 9, 1)
plt.imshow(img.cpu().squeeze(), cmap='gray')
plt.title(f'原图\n标签: {true_label}')
plt.axis('off')

# 显示前16个feature map
for i in range(16):
    plt.subplot(2, 9, i+2)
    plt.imshow(relu_output[0, i, :, :].cpu().numpy(), cmap='gray')
    plt.title(f'FM {i+1}')
    plt.axis('off')

plt.tight_layout()
plt.savefig('task5_feature_maps.png', dpi=300)
plt.close()
print("✅ Feature map已保存: task5_feature_maps.png")
print("说明：不同的feature map对图像的不同区域有强响应，分别提取了图像的不同特征。")
print("例如，有的feature map响应数字的边缘，有的响应数字的内部纹理，有的响应特定的笔画方向。")

# ===================== 任务6：错误分类样本分析 =====================
print("\n" + "="*50)
print("任务6：错误分类样本分析")
print("="*50)

# 找出所有错误分类的样本
model.eval()
wrong_images = []
wrong_true_labels = []
wrong_pred_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        
        # 找出错误预测的样本
        wrong_mask = predicted != labels
        wrong_images.extend(images[wrong_mask].cpu())
        wrong_true_labels.extend(labels[wrong_mask].cpu().numpy())
        wrong_pred_labels.extend(predicted[wrong_mask].cpu().numpy())

# 显示前16个错误分类样本
plt.figure(figsize=(12, 12))
for i in range(min(16, len(wrong_images))):
    plt.subplot(4, 4, i+1)
    plt.imshow(wrong_images[i].squeeze(), cmap='gray')
    plt.title(f'True: {wrong_true_labels[i]}\nPred: {wrong_pred_labels[i]}', fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.savefig('task6_wrong_samples.png', dpi=300)
plt.close()
print(f"✅ 错误分类样本已保存: task6_wrong_samples.png")
print(f"总错误样本数: {len(wrong_images)}")
print(f"错误率: {len(wrong_images)/len(test_dataset)*100:.2f}%")

# 统计最容易混淆的类别
confusion_pairs = {}
for true, pred in zip(wrong_true_labels, wrong_pred_labels):
    pair = (true, pred)
    confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1

# 按混淆次数排序
sorted_confusion = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
print("\n最容易混淆的类别对（真实类别→预测类别）:")
for i in range(min(10, len(sorted_confusion))):
    (true, pred), count = sorted_confusion[i]
    print(f"{true} → {pred}: {count}次")

print("\n错误原因分析:")
print("1. 数字形状相似：如3和5、4和9、7和1等")
print("2. 手写风格差异：部分数字书写不规范，与训练集中的样本差异较大")
print("3. 图像质量问题：部分图像模糊、有噪声或笔画断裂")

print("\n改进建议:")
print("1. 数据增强：增加旋转、平移、缩放、噪声等数据增强操作")
print("2. 模型改进：增加网络深度、使用Batch Normalization、调整Dropout率")
print("3. 训练优化：使用学习率调度器、增加训练轮数、尝试不同的优化器")

# ===================== 任务7：混淆矩阵 =====================
print("\n" + "="*50)
print("任务7：混淆矩阵绘制与分析")
print("="*50)

# 计算混淆矩阵
cm = confusion_matrix(labels_base, preds_base)

# 绘制混淆矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('task7_confusion_matrix.png', dpi=300)
plt.close()
print("✅ 混淆矩阵已保存: task7_confusion_matrix.png")

print("\n混淆矩阵分析:")
print("1. 对角线元素：代表模型正确分类的样本数量")
print("2. 非对角线元素：代表模型错误分类的样本数量")
print("3. 混淆最严重的类别对：")
for i in range(min(5, len(sorted_confusion))):
    (true, pred), count = sorted_confusion[i]
    print(f"   真实类别{true}被错误预测为类别{pred}，共{count}次")

print("\n🎉 所有实验任务已完成！所有图表已保存到当前目录。")