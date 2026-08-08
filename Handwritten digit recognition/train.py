import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

train_dataset = datasets.MNIST(root=r"D:\remotions\datasets", #路径前加r 可以避免反斜杠被当作转义字符
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)
test_dataset = datasets.MNIST(root=r"D:\remotions\datasets",
                               train=False,
                               transform=transforms.ToTensor(),
                               download=True)

batch_size = 64  #设置batch大小为64 即一次从训练集里取出64张图片

train_loader = DataLoader(dataset=train_dataset,
                           batch_size=batch_size,
                           shuffle=True)  #每轮都打乱一下数据 让每次组成的batch不完全一样
test_loader = DataLoader(dataset=test_dataset,
                           batch_size=batch_size,
                           shuffle=False)

#先拿出一个batch 看看图片和标签的形状对不对
images, labels = next(iter(train_loader))

print("训练集样本数量：", len(train_dataset))
print("测试集样本数量：", len(test_dataset))
print("一个批次的图片形状：", images.shape)
print("一个批次的标签形状：", labels.shape)
print("前10个标签：", labels[:10])