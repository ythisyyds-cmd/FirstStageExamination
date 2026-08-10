import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader

train_dataset = datasets.MNIST(root="./datasets",  #数据集会自动下载到datasets文件夹
                               train=True,
                               transform=transforms.ToTensor(),
                               download=True)
test_dataset = datasets.MNIST(root="./datasets",
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

#取出每张图片正确类别的概率 再计算这个batch的平均交叉熵损失
def cross_entropy(p_output, labels):
    rows = labels.shape[0]
    loss = 0
    for i in range(rows):                     #逐张取出真实数字对应的预测概率 并把每张图片的损失加起来
        label = labels[i].item()              #取第i张的真实数字
        temp_p = p_output[i, label]           #取第i张对应数字下的预测概率 训练得越好这个概率应该越接近1

        loss += -torch.log(temp_p + 1e-12)    #把真实数字这一类看作y=1 此时-[y*log(p)+(1-y)*log(1-p)]可以化简为-log(p) 多分类交叉熵只需取出真实数字对应的这一项
                                              #加一个很小的数当保险 防止概率太小时计算机把它当成0

    return loss / rows                        #用总损失除以图片数量 得到这个batch的平均损失

#根据前向传播保存的结果 计算四个参数的梯度
def backward(x, hidden_output, p_output, labels):
    rows = labels.shape[0]

    grad_z2 = p_output.clone()                    #从loss到z2 其他类别的梯度是p 真实类别位置的梯度是p-1
    grad_z2[torch.arange(rows), labels] -= 1

    grad_w2 = hidden_output.T @ grad_z2 / rows    #计算隐藏层到输出层的权重和偏置梯度
    grad_b2 = grad_z2.mean(dim=0)

    grad_hidden = grad_z2 @ w2.T                  #把梯度传回隐藏层 再乘上sigmoid的导数
    grad_z1 = grad_hidden * hidden_output * (1 - hidden_output)

    grad_w1 = x.T @ grad_z1 / rows                #计算输入层到隐藏层的权重和偏置梯度
    grad_b1 = grad_z1.mean(dim=0)

    return grad_w1, grad_b1, grad_w2, grad_b2    


lr = 0.1       #学习率设为0.1
epochs = 10    #训练5轮后发现有进步空间 故调整为10轮

#正式训练网络
for epoch in range(epochs):
    total_loss = 0

    for images, labels in train_loader:
        x, hidden_output, p_output = forward(images)          #前向传播并计算损失
        loss = cross_entropy(p_output, labels)

        grad_w1, grad_b1, grad_w2, grad_b2 = backward(        #反向传播计算四个参数的梯度
            x, hidden_output, p_output, labels
        )

        w1 -= lr * grad_w1                                    #更新权重和偏置
        b1 -= lr * grad_b1
        w2 -= lr * grad_w2
        b2 -= lr * grad_b2

        total_loss += loss.item() * labels.shape[0]           #把这个batch中所有图片的损失加到总损失中

    average_loss = total_loss / len(train_dataset)            #计算这一轮所有训练图片的平均损失
    print(f"第{epoch + 1}轮训练 平均损失：{average_loss:.4f}")


#用测试集来计算识别准确率
correct = 0                                             
total = 0

for images, labels in test_loader:

    x, hidden_output, p_output = forward(images)        #进行前向传播

    predictions = p_output.argmax(dim=1)                #取10个概率中最大值的位置作为预测数字

    correct += (predictions == labels).sum().item()     #统计预测正确的图片数量

    total += labels.shape[0]

accuracy = correct / total * 100
print(f"测试集准确率：{accuracy:.2f}%")