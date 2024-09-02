import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms, datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tqdm import tqdm
from train_Resnet50 import ResNet50
from train_ResNet18 import ResNet18
from train_Densenet121 import DenseNet121
import random
import matplotlib.pyplot as plt
torch.backends.cudnn.enabled = False
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

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

num_epochs=15
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet18(num_classes=4).to(device)
# 损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)  # 随机梯度下降优化器
num_epochs = 100
learning_rate = 0.001
M=3
alpha=0.01


    # 训练模型
torch.autograd.set_detect_anomaly(True)
def train(model,criterion,optimizer,train_loader,test_loader):
    choices = ['resnet50', 'resnet18', 'densenet']

    for epoch in range(int(num_epochs)):
        resnet50 = ResNet50(num_classes=10).to(device)
        resnet50.load_state_dict(torch.load('./Best_model_Resnet50_weights.pth'))
        for param in resnet50.parameters():
            param.requires_grad = False
        densenet = DenseNet121(num_classes=10).to(device)
        densenet.load_state_dict(torch.load('./Best_model_DenseNet121_weights.pth'))
        for param in densenet.parameters():
            param.requires_grad = False
        resnet18 = ResNet18(num_classes=4).to(device)
        resnet18.load_state_dict(torch.load('./Best_model_ResNet18_weights.pth'))
        for param in resnet18.parameters():
            param.requires_grad = False
        pert = torch.FloatTensor(*(32,1,224,224)).uniform_(-alpha, alpha).to(device)
        # pert带有梯度
        pert.requires_grad_()
        flag = random.choice(choices)
        flag = 'resnet'
        for i, (images, labels) in enumerate(tqdm(train_dataloader, desc="Training")):
            images, labels = images.to(device), labels.to(device)
            # 为输入数据增加对抗性扰动pert
            if flag=='resnet50':
                out = resnet50(images + pert)
            elif flag=='resnet18':
                out = resnet18(images + pert)
            else:
                out = densenet(images + pert)
            # 因为loss的梯度一直是累加的，所以每个step贡献1/M的grad值
            if i == 100:
                break
            loss = criterion(out, labels)

            # 每个epoch分为M个step，M个loss的grad进行累加，得到最终的loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 根据pert的grad来更新pert，alpha可以看作是pert的学习率
            pert_data = pert.detach() + alpha * torch.sign(pert.grad.detach())
            pert.data = pert_data.data
            # pert梯度grad清零
            pert.grad[:] = 0
            torch.cuda.empty_cache()
        torch.save(pert,'./noise/'+str(epoch)+'.pt')




if __name__ == '__main__':
    # 准确率列表
    train(model, criterion, optimizer, train_dataloader, test_dataloader)
