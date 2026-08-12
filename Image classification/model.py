import torch
import torch.nn as nn

#让输入经过两次卷积后再与原来的输入相加 做一个简单的残差块
class ResBlock(nn.Module):
    def __init__(self, in_channelNum,out_channelNum,default_stride = 1):    #步长默认为1
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channelNum,                   #第一层卷积负责改变通道数 必要时也会把图片缩小
                               out_channels=out_channelNum,         
                               kernel_size=3,
                               stride=default_stride,
                               padding=1,
                               bias=False)
        
        self.bn1 = nn.BatchNorm2d(num_features=out_channelNum)
        self.relu = nn.ReLU(inplace=False)

        self.conv2 = nn.Conv2d(in_channels=out_channelNum,                  #第二层卷积保持通道数和图片大小不变
                               out_channels=out_channelNum,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(num_features=out_channelNum)

        channel_ischanged = False
        if in_channelNum != out_channelNum:                                 #判断输入和输出通道数是否不同
            channel_ischanged = True

        size_ischanged = False
        if default_stride != 1:                                             #判断图片大小是否发生变化
            size_ischanged = True

        self.needShortcut = channel_ischanged or size_ischanged

        if self.needShortcut:                                               #只要通道数或图片大小改变 就需要调整原来的输入
            self.shortcutConv = nn.Conv2d(in_channels=in_channelNum,        #使用1×1卷积把原输入调整成和卷积结果相同的形状
                                          out_channels=out_channelNum,
                                          kernel_size=1,
                                          stride=default_stride,
                                          padding=0,
                                          bias=False)
            
            self.shortcutBn = nn.BatchNorm2d(num_features=out_channelNum)


    def forward(self,x):
        temp = x                                #暂存原来的输入
        x = self.conv1(x)                       #第一次卷积
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)                       #第二次卷积
        x = self.bn2(x)

        if self.needShortcut:                   #调整原输入的通道数和图片大小 不然形状不同下一步没法相加
            temp = self.shortcutConv(temp)      
            temp = self.shortcutBn(temp)

        x = x + temp                            #把原来的输入加回来
        x = self.relu(x)

        return x


#搭建ResNet18网络
class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()

        self.cifarchanger = nn.Conv2d(in_channels=3,        #处理CIFAR-10 把图片的通道数从3变成64
                                      out_channels=64,
                                      kernel_size=3,
                                      stride=1,
                                      padding=1,
                                      bias=False)
        
        self.startBn = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=False)

        self.block1 = ResBlock(in_channelNum=64,           #构造两个形状不变的残差块
               out_channelNum=64,
               default_stride=1)
        self.block2 = ResBlock(in_channelNum=64,
               out_channelNum=64,
               default_stride=1)
        self.block3 = ResBlock(in_channelNum=64,           #第二组第一个残差块负责改变通道数和图片大小
                               out_channelNum=128,
                               default_stride=2)
        
        self.block4 = ResBlock(in_channelNum=128,          #第二个残差块保持形状不变
                               out_channelNum=128,
                               default_stride=1)
        self.block5 = ResBlock(in_channelNum=128,          #第三组第一个残差块改变通道数和图片大小
                               out_channelNum=256,
                               default_stride=2)
        
        self.block6 = ResBlock(in_channelNum=256,          #第二个残差块保持形状不变
                               out_channelNum=256,
                               default_stride=1)

        self.block7 = ResBlock(in_channelNum=256,          #第四组第一个残差块改变通道数和图片大小
                               out_channelNum=512,
                               default_stride=2)
        
        self.block8 = ResBlock(in_channelNum=512,          #第二个残差块保持形状不变
                               out_channelNum=512,
                               default_stride=1)
        
        self.avgPool = nn.AdaptiveAvgPool2d(output_size=(1, 1))  #做平均池化

        self.featurechanger = nn.Linear(in_features=512,         #把512个特征转换成10个类别的结果
                            out_features=10)

    def forward(self,x):                       #让图片经过ResNet18 得到10个类别的输出
        x = self.cifarchanger(x)
        x = self.startBn(x)
        x = self.relu(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.avgPool(x)                   #形状从[batch,512,4,4]变成[batch,512,1,1]

        batch_num = x.shape[0]                #取出这个batch实际有多少张图片
        x = x.reshape(batch_num, 512)         #展平为每张图片512个特征

        x = self.featurechanger(x)            #得到对10个类别的输出
        return(x)


