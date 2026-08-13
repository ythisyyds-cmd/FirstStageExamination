import torch
from torchvision import transforms
from PIL import Image
from model import ResNet18


class_names = [                                         #CIFAR-10里的10个类名

    "飞机",
    "汽车",
    "鸟",
    "猫",
    "鹿",
    "狗",
    "青蛙",
    "马",
    "船",
    "卡车"
]


#把输入图片处理成网络能接收的形式
image_transform = transforms.Compose(transforms=[
    transforms.Resize(size=(32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5, 0.5, 0.5),
                         std=(0.5, 0.5, 0.5))
])


#建立ResNet18并读取训练好的参数
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model = ResNet18()

model_parameters = torch.load(
    "./Image classification/resnet18_cifar10.pth",
    map_location=device,
    weights_only=True
)

model.load_state_dict(model_parameters)
model = model.to(device)
model.eval()


#传入一张图片的路径 返回网络预测的类别
def predict_image(image_path):
    image = Image.open(image_path)                 #打开图片
    image = image.convert("RGB")                   #保证图片有红绿蓝3个通道
    image = image_transform(image)                 #调整图片大小并进行标准化

    image = image.unsqueeze(dim=0)                 #增加batch这一维
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)                     #得到图片对10个类别的输出
        prediction = torch.argmax(outputs, dim=1)  #找到输出最大值对应的类别编号

    prediction_num = prediction.item()
    prediction_name = class_names[prediction_num]

    return prediction_name


#用一张图片测试识别函数
image_path = "./Image classification/test_image.jpg"

prediction_result = predict_image(image_path)

print("预测类别：", prediction_result)