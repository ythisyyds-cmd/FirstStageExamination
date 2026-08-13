import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from model import ResNet18

#对训练集做数据增强
train_transform = transforms.Compose(transforms=[      #训练集先随机改变图片 再转换成张量并进行标准化
    transforms.RandomCrop(size=32,
                          padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5),
                         std=(0.5, 0.5, 0.5))
])



test_transform = transforms.Compose(transforms=[       #测试集不随机变化 只转换成张量并进行标准化
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5),
                         std=(0.5, 0.5, 0.5))
])


#下载并读取CIFAR-10训练集和测试集
train_dataset = datasets.CIFAR10(root="./datasets",
                                 train=True,
                                 transform=train_transform,
                                 download=True)

test_dataset = datasets.CIFAR10(root="./datasets",
                                train=False,
                                transform=test_transform,
                                download=True)


batch_size = 128                                      #batch的大小为128 一次取出128张图片

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shuffle=False)


#有显卡时使用显卡训练 否则使用CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


#把之前写好的ResNet18放到训练设备中
model = ResNet18()
model = model.to(device)


#损失函数用交叉熵 优化器用SGD
loss_function = nn.CrossEntropyLoss()

optimizer = optim.SGD(params=model.parameters(),
                      lr=0.1,
                      momentum=0.9,
                      weight_decay=0.0005)


epochs = 20                                          #训练20轮


#正式开始训练网络
model.train()

for epoch in range(epochs):
    total_loss = 0

    for images, labels in train_loader:
        images = images.to(device)                   #把图片和标签放到显卡中
        labels = labels.to(device)                   

        optimizer.zero_grad()                        #清零梯度

        outputs = model(images)                      #让图片经过ResNet18
        loss = loss_function(outputs, labels)        #计算损失

        loss.backward()                              #计算所有参数的梯度
        optimizer.step()                             #根据梯度更新网络参数

        total_loss += loss.item() * labels.shape[0]  #累加这个batch里所有图片的损失

    average_loss = total_loss / len(train_dataset)
    print(f"第{epoch + 1}轮训练 平均损失：{average_loss:.4f}")


#用测试集来检查网络的识别准确率
model.eval()

correct_num = 0
total_num = 0

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)                         #得到每张图片对10个类别的输出
        predictions = torch.argmax(outputs, dim=1)      #取最大值所在的位置作为预测的类别

        correct_num += (predictions == labels).sum().item()
        total_num += labels.shape[0]

accuracy = correct_num / total_num * 100
print(f"测试集准确率：{accuracy:.2f}%")


#保存训练好的网络参数
torch.save(model.state_dict(), "./Image classification/resnet18_cifar10.pth")
print("模型参数已保存")