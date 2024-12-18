import torch
import torch.nn as nn
import torch.quantization
from torchinfo import summary


# 查看浮点模型的详细信息

# 定义一个简单的全连接神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        # 添加 QuantStub 和 DeQuantStub
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()

        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        # 在输入时进行量化
        x = self.quant(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        # 在输出时进行反量化
        x = self.dequant(x)
        return x


# 创建模型实例
model = SimpleNet()
summary(model, input_size=(64, 784))
# 模拟训练好的模型（这里省略了实际训练过程）
# model.load_state_dict(torch.load('pretrained_model.pth'))

# 设置为评估模式
model.eval()

# 配置量化设置
model.qconfig = torch.quantization.get_default_qconfig('fbgemm')  # 选择量化配置

# 插入量化操作
torch.quantization.prepare(model, inplace=True)


# 校准模型
def calibrate(model, data_loader):
    model.eval()
    with torch.no_grad():
        for data, _ in data_loader:
            model(data)


# 使用模拟的数据加载器进行校准
dummy_data_loader = [(torch.randn(64, 784), torch.randint(0, 10, (64,))) for _ in range(10)]  # 假设输入是28x28图像展平后的向量
calibrate(model, dummy_data_loader)

# 转换为量化模型
torch.quantization.convert(model, inplace=True)

# 保存量化后的模型
torch.save(model.state_dict(), 'quantized_model.pth')


# 评估量化后的模型
def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            output = model(data)  # 输入和输出会自动进行量化和反量化
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')


# 使用模拟的数据加载器进行评估
dummy_test_loader = [(torch.randn(64, 784), torch.randint(0, 10, (64,))) for _ in range(10)]
evaluate(model, dummy_test_loader)
summary(model, input_size=(64, 784))
