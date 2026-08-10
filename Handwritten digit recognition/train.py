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

input_size = 784   #一张图片展开后有784个像素
hidden_size = 128  #隐藏层先设128个神经元
output_size = 10   #输出对应0到9这10个数字

#初始化权重和偏置
w1 = torch.randn(input_size, hidden_size) * 0.01  #权重不能全部设成0 所以先使用比较小的随机数
b1 = torch.zeros(hidden_size)                     #偏置先从0开始 后续再根据梯度更新
w2 = torch.randn(hidden_size, output_size) * 0.01  
b2 = torch.zeros(output_size)

#sigmoid会分别处理张量中的每个数 把结果压缩到0和1之间
def sigmoid(x):
    temp = 1 + torch.exp(-x)
    return 1 / temp

#softmax可以把输出层的10个结果转换成概率 每一行的概率加起来等于1
def softmax(x):
    l_max = x.max(dim=1,keepdim=True).values  #取每行最大值
    x = x - l_max                             #每一行减去自己的最大值 防止指数过大
    exp_x = torch.exp(x)                      #分别求每个元素的指数
    sum_exp = exp_x.sum(dim=1,keepdim=True)   #求每一行的指数总和
    return exp_x / sum_exp  

#让一个batch的图片经过隐藏层和输出层 得到对0到9的预测概率
def forward(images):
    rows = images.shape[0]                    #确定取出的这个batch实际有多少张图片 然后把每张图片拉平成784个像素
    x = images.reshape(rows,input_size)
    hidden_input = x @ w1 + b1                #计算输入层到隐藏层的结果 然后经过sigmoid
    hidden_output = sigmoid(hidden_input)
    output_input = hidden_output @ w2 + b2    #再从隐藏层计算到输出层 最后用softmax转换成概率
    p_output = softmax(output_input)
    return x,hidden_output,p_output           #返回这些结果 后面计算损失和梯度时还会用到


#先取出一个batch 检查前向传播各部分的形状是否正确
images, labels = next(iter(train_loader))
x, hidden_output, p_output = forward(images)
print("图片展开后的形状：", x.shape)
print("隐藏层输出形状：", hidden_output.shape)
print("最终输出形状：", p_output.shape)
print("第一张图片的概率总和：", p_output[0].sum())