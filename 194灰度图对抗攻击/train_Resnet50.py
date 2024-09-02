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

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def ResNet50(num_classes):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)

if __name__ == '__main__':
    # 如果cuda可用，就用cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNet50(num_classes=10).to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # 随机梯度下降优化器


    import random
    import numpy as np
    import matplotlib.pyplot as plt

    # 设置随机种子以便复现性
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)


    # 创建训练集和测试集的数据加载器
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 训练循环
    train_losses = []
    test_losses = []
    accuracies = []

    num_epochs = 15
    bset_accuracy=0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(tqdm(train_dataloader, desc="Training")):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_dataloader)
        train_losses.append(train_loss)

        # 在测试集上评估模型
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

        accuracies.append(accuracy)


        print(f"Epoch {epoch + 1}, Train Loss: {train_loss}, Test Loss: {test_loss}")
        if bset_accuracy<accuracy:
            bset_accuracy = accuracy
            torch.save(model.state_dict(), './Best_model_Resnet50_weights.pth')
        print(f"Accuracy: {accuracy}, Bset_accuracy: {bset_accuracy}")
        print(f"Accuracy: {accuracy}")

    torch.save(model.state_dict(),'./model_Resnet50_weights.pth')

# # 检查模型结构
# if __name__ == "__main__":
#     model = resnet50(num_classes=10)
#     print(model)
#
#     # 生成一个示例输入
#     sample_input = torch.randn(1, 1, 224, 224)  # Batch size = 1, Gray image (1 channel), 28x28
#     output = model(sample_input)
#     print("Output shape:", output.shape)  # 期望输出形状: [1, 10]


