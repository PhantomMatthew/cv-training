"""
车辆品牌识别模型训练脚本
Car Brand Classification Training Script
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


# 10个车辆品牌类别的映射
CAR_BRANDS = [
    "Toyota_Camry",      # 1 - 丰田_凯美瑞
    "Toyota_Corolla",    # 2 - 丰田_卡罗拉
    "Toyota_Corolla_EX", # 3 - 丰田_花冠
    "Buick_LaCrosse",    # 4 - 别克_君越
    "Volkswagen_Magotan",# 5 - 大众_迈腾
    "Audi_A4",           # 6 - 奥迪_A4
    "Nissan_Sylphy",     # 7 - 日产_轩逸
    "Nissan_Tiida",      # 8 - 日产_骐达
    "Honda_Accord",      # 9 - 本田_雅阁
    "Ford_Focus"         # 10 - 福特_福克斯
]


class CarBrandDataset(Dataset):
    """车辆品牌数据集"""
    def __init__(self, index_file, image_base_dir, transform=None):
        """
        Args:
            index_file: 索引文件路径 (re_id_1000_train.txt 或 re_id_1000_test.txt)
            image_base_dir: 图像根目录 (CV-车辆检测/image/)
            transform: 图像变换
        """
        self.image_base_dir = image_base_dir
        self.transform = transform
        self.samples = []

        with open(index_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    # 格式: class_folder\License_x\filename.jpg
                    # class_folder (1-10) 对应的标签是 (0-9)
                    parts = line.replace('/', '\\').split('\\')
                    if len(parts) >= 3:
                        class_folder = parts[0]
                        img_path = os.path.join(image_base_dir, line.replace('/', os.sep))
                        label = int(class_folder) - 1  # 转换为 0-9 的标签
                        self.samples.append((img_path, label))

        print(f"Loaded {len(self.samples)} samples from {index_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # 返回一个空白图像
            image = Image.new('RGB', (224, 224), color='white')

        if self.transform:
            image = self.transform(image)

        return image, label


def get_model(num_classes=10, pretrained=True):
    """创建基于ResNet18的车辆分类模型"""
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

    # 修改最后的全连接层
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )

    return model


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': running_loss / (pbar.n + 1),
            'acc': 100. * correct / total
        })

    return running_loss / len(train_loader), 100. * correct / total


def validate(model, val_loader, criterion, device):
    """验证模型"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(val_loader), 100. * correct / total


def test(model, test_loader, device):
    """测试模型"""
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # 计算每个类别的准确率
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1

    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Overall Accuracy: {100. * correct / total:.2f}%")

    print("\nPer-class Accuracy:")
    for i in range(10):
        if class_total[i] > 0:
            acc = 100. * class_correct[i] / class_total[i]
            print(f"  {CAR_BRANDS[i]:25s}: {acc:6.2f}% ({class_correct[i]}/{class_total[i]})")
    print("="*60 + "\n")

    return 100. * correct / total


def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)

    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 数据路径
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "CV-车辆检测", "image")
    train_index = os.path.join(base_dir, "CV-车辆检测", "re_id_1000_train.txt")
    test_index = os.path.join(base_dir, "CV-车辆检测", "re_id_1000_test.txt")

    # 数据变换
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 创建数据集
    print("Loading datasets...")
    full_train_dataset = CarBrandDataset(train_index, data_dir, transform=train_transform)
    test_dataset = CarBrandDataset(test_index, data_dir, transform=val_test_transform)

    # 划分训练集和验证集 (80% train, 20% val)
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # 为验证集设置正确的变换
    val_dataset.dataset.transform = val_test_transform

    print(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}, Test size: {len(test_dataset)}")

    # 创建数据加载器
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    # 创建模型
    print("\nCreating model...")
    model = get_model(num_classes=10, pretrained=True)
    model = model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # 训练参数
    num_epochs = 50
    best_val_acc = 0.0
    patience = 10
    patience_counter = 0

    print("\nStarting training...")
    print("="*60)

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        print("-"*60)

        # 训练
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)

        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # 更新学习率
        scheduler.step()

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(base_dir, 'best_model.pth'))
            print(f"*** New best model saved! Val Acc: {val_acc:.2f}% ***")
        else:
            patience_counter += 1

        # 早停
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print("="*60)

    # 加载最佳模型进行测试
    print("\nLoading best model for testing...")
    checkpoint = torch.load(os.path.join(base_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    # 测试
    test_acc = test(model, test_loader, device)

    # 检查是否达到80%的要求
    print("\n" + "="*60)
    if test_acc >= 80.0:
        print(f"SUCCESS! Test accuracy ({test_acc:.2f}%) meets requirement (>=80%)")
    else:
        print(f"WARNING! Test accuracy ({test_acc:.2f}%) is below requirement (>=80%)")
        print("Please tune hyperparameters and retrain.")
    print("="*60)

    # 保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'test_acc': test_acc,
        'car_brands': CAR_BRANDS
    }, os.path.join(base_dir, 'final_model.pth'))

    print(f"\nModel saved to: {os.path.join(base_dir, 'final_model.pth')}")


if __name__ == '__main__':
    main()
