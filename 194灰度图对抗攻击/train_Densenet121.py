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


# 定义Bottleneck Layer
class BottleneckLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size=4, drop_rate=0.0):
        super(BottleneckLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size *
                               growth_rate, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        if self.drop_rate > 0:
            out = nn.functional.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], 1)


# (Dense Block)
class DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList([
            BottleneckLayer(
                num_input_features + i * growth_rate,
                growth_rate,
                bn_size,
                drop_rate
            ) for i in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


# (Transition Layer)
class TransitionLayer(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(TransitionLayer, self).__init__()
        self.bn = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features,
                              kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(self.relu(self.bn(x)))
        x = self.pool(x)
        return x


# DenseNet121mod
class DenseNet121(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=10):
        super(DenseNet121, self).__init__()

        # init
        self.features = nn.Sequential(
            nn.Conv2d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Dense Block and Transition Layer
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate)
            self.features.add_module('denseblock{}'.format(i + 1), block)
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = TransitionLayer(num_features, num_features // 2)
                self.features.add_module('transition{}'.format(i + 1), trans)
                num_features = num_features // 2

        #BN
        self.features.add_module('bn', nn.BatchNorm2d(num_features))
        self.classifier = nn.Linear(num_features, num_classes)

    def forward(self, x):
        features = self.features(x)
        out = nn.ReLU(inplace=True)(features)
        out = nn.AdaptiveAvgPool2d((1, 1))(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out



if __name__ == '__main__':
    # 如果cuda可用，就用cuda
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DenseNet121(num_classes=10).to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # 随机梯度下降优化器


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

    num_epochs = 3
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
            torch.save(model.state_dict(), 'Best_model_DenseNet121_weights.pth')
        print(f"Accuracy: {accuracy}, Bset_accuracy: {bset_accuracy}")

    torch.save(model.state_dict(),'./model_DenseNet121_weights.pth')

# # 检查模型结构
# if __name__ == "__main__":
#     model = DenseNet121(num_classes=10)
#     print(model)
#
#     # 生成一个示例输入
#     sample_input = torch.randn(1, 1, 224, 224)  # Batch size = 1, Gray image (1 channel), 28x28
#     output = model(sample_input)
#     print("Output shape:", output.shape)  # 期望输出形状: [1, 10]

