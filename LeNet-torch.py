from __future__ import print_function
from torch import optim
import torch as t

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
# import matplotlib.pyplot as plt
import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
show = ToPILImage()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize(mean, std)
])


# 训练集
train_set = tv.datasets.CIFAR10(
    root='/Users/weiyangbin/Downloads/cifar-100-python/',
    train=True,
    download=True,
    transform=transform
)


train_loader = t.utils.data.DataLoader(
    train_set,
    batch_size=4,
    shuffle=True,
    num_workers=2
)

test_set = tv.datasets.CIFAR10(
    root='Users/weiyangbin/Downloads/cifar-100-python',
    train=False,
    download=True,
    transform=transform
)

test_loader = t.utils.data.DataLoader(
    test_set,
    batch_size=4,
    shuffle=False,
    num_workers=2
)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# (data, label) = train_set[100]
# show((data + 1) / 2).resize((100, 100))
# cnt = 0
# to_plt_image = transforms.ToPILImage()
# for image, label in train_loader:
#     if cnt > 2:
#         break
#     print(label)
#
#     img = to_plt_image(image[0])
#     img.show()
#
#     plt.imshow(img)
#     plt.show()
#     cnt += 1


# data_iter = iter(train_loader)
# images, label = data_iter.next()

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


net = Net()
print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

t.set_num_threads(8)
for epoch in range(5):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):

        # 输入数据
        inputs, labels = data

        # 梯度清零
        optimizer.zero_grad()

        # 前向传播，方向传播
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()

        # 更新参数
        optimizer.step()

        # 打印log信息
        # loss是一个scalar， 需要loss.item()来获取数值，不能用loss[0]
        running_loss += loss.item()
        if i % 2000 == 1999:
            print('[%d, %5d] loss: %.3f'\
                  % (epoch+1, i+1, running_loss / 2000))
            running_loss = 0.0

print('Finish Training')

dataiter = iter(test_loader)
images, labels = dataiter.next()
print('实际label： ', ' '.join(\
    '%08s' % classes[labels[j]] for j in range(4)))


outputs = net(images)
_,  predicted = t.max(outputs.data, 1)

print('预测结果：', ' '.join(\
    '%5s' % classes[predicted[j]] for j in range(4)))

correct = 0
total = 0

with t.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
        _, predicted = t.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

print('10000张测试集中准确率为： %d %%' % (100 * correct / total))
