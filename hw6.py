import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as dset
from torchvision import datasets
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm

# 设置设备为GPU（若可用）否则为CPU
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# list中为数据转化的方法，将按照list中元素的顺序执行方法来处理数据
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,)),]
)

# 加载MNIST数据集，训练集和测试集
trainSet = datasets.MNIST(root='./data', download=True, train=True, transform=transform)
testSet = datasets.MNIST(root='./data', download=True, train=False, transform=transform)
trainLoader = dset.DataLoader(trainSet, batch_size=64, shuffle=True)
testLoader = dset.DataLoader(testSet, batch_size=64, shuffle=False)

# 通用感应器网络（全连接层+relu）
class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def name(self):
        return "MLP"


# 最早的卷积神经网络之一，两层卷积+池化，全连接
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def name(self):
        return "LeNet"

# 实例化LeNet模型并移动到指定设备
model = LeNet().to(device)

# 训练轮数
epochs = 3
# 学习率
lr = 0.002
# 损失函数使用交叉熵损失
criterion = nn.CrossEntropyLoss()
# 优化器使用SGD，带动量
optimizer = optim.SGD(model.parameters(), lr=0.002, momentum=0.9)

# 训练过程
for epoch in range(epochs):
    running_loss = 0.0

    for idx, data in enumerate(trainLoader):
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # 累加损失并定期打印
        running_loss += loss.item()
        if idx % 100 == 99 or idx+1 == len(trainLoader):
            print('[%d/%d, %d/%d] loss: %.3f' % (epoch+1, epochs, idx+1, len(trainLoader), running_loss/2000))
            running_loss = 0.0  # 重置运行损失

print('Training Finished.')

# 测试整体准确率
correct = 0
total = 0

with torch.no_grad():
    for data in testLoader:
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        # 累加总样本数
        total += labels.size(0)
        # 累加正确预测数
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (100*correct / total))

# 按类别计算准确率
class_correct = [0 for i in range(10)]
class_total = [0 for i in range(10)]

with torch.no_grad():
    for data in testLoader:
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(len(labels)):  # 遍历批次中的每个样本
            label = labels[i]
            # 累加该类别的正确预测数
            class_correct[label] += c[i].item()
            # 累加该类别的总样本数
            class_total[label] += 1

for i in range(10):
    print('Accuracy of %d: %.3f' % (i, (class_correct[i]/class_total[i]) if class_total[i] > 0 else 0))

# 保存模型参数
torch.save(model.state_dict(), model.name()+".pt")