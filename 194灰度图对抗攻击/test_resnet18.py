import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm


transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 将图像大小调整为 64x64
    transforms.Grayscale(num_output_channels=1),  # 将图像转换为灰度图像
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize((0.5,), (0.5,))  # 归一化图像张量，只需一个均值和标准差用于单通道
])

# 数据加载
train_data_dir = './data/train'  # 数据目录的路径
test_data_dir = './data/test'  # 数据目录的路径
batch_size = 32  # 批量大小
train_dataset = torchvision.datasets.ImageFolder(root=train_data_dir, transform=transform)  # 创建数据集
test_dataset = torchvision.datasets.ImageFolder(root=test_data_dir, transform=transform)  # 创建数据集


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def ResNet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes)


if __name__ == '__main__':
    # 如果cuda可用，就用cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet18(num_classes=4).to(device)
    model.load_state_dict(torch.load('./Best_model_ResNet18_weights.pth'))
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr=0.001, )  # 随机梯度下降优化器


    import random
    import numpy as np
    import matplotlib.pyplot as plt

    # 设置随机种子以便复现性
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # 创建训练集和测试集的数据加载器
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 训练循环
    train_losses = []
    test_losses = []
    accuracies = []

    num_epochs = 5
    bset_accuracy=0
    model.eval()
    running_loss = 0.0
    test_predictions = []
    test_labels = []

    for inputs, labels in tqdm(test_dataloader, desc="Evaluating"):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()

        _, predicted = torch.max(outputs.data, 1)
        test_predictions.extend(predicted.tolist())
        test_labels.extend(labels.tolist())

    test_loss = running_loss / len(test_dataloader)
    test_losses.append(test_loss)

    # 计算指标
    accuracy = accuracy_score(test_labels, test_predictions)
    print('accuracy: ',accuracy)
