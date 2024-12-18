import torch
import torch.nn as nn
import torch.quantization
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# 1. 定义模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # 卷积层 + 批归一化 + 激活函数
        self.conv_bn_relu1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv_bn_relu2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # 全连接层
        self.fc1 = nn.Linear(128 * 8 * 8, 10)

    def forward(self, x):
        x = self.conv_bn_relu1(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv_bn_relu2(x)
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


# 2. 配置混合精度静态量化
def configure_mixed_precision_static_quantization(model, backend='fbgemm'):
    # 设置量化引擎
    torch.backends.quantized.engine = backend

    # 为卷积层和全连接层设置量化配置
    model.qconfig = torch.quantization.get_default_qconfig(backend)

    # 为特定层设置不同的量化配置
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            # 对卷积层和全连接层使用低精度（int8）
            module.qconfig = torch.quantization.get_default_qconfig(backend)
        elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.ReLU):
            # 对批归一化层和激活函数层保持高精度（float32）
            module.qconfig = None  # 不进行量化

    # 准备静态量化
    model = torch.quantization.prepare(model)

    return model


# 3. 校准模型（收集统计信息）
def calibrate_model(model, data_loader):
    model.eval()  # 确保模型处于评估模式
    with torch.no_grad():
        for inputs, _ in data_loader:
            model(inputs)


# 4. 转换为真实量化模型
def convert_to_quantized_model(model):
    return torch.quantization.convert(model, inplace=False)


# 5. 评估模型性能
def evaluate_model(model, test_loader):
    model.eval()  # 确保模型处于评估模式
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Test Accuracy: {100 * correct / total:.2f}%')
    model.train()  # 评估完成后恢复训练模式


# 6. 准备数据集
def prepare_data(batch_size=64):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, test_loader


# 7. 主函数
if __name__ == '__main__':
    # 准备数据集
    train_loader, test_loader = prepare_data()

    # 创建模型实例
    model = SimpleCNN()

    # 加载预训练模型（假设你有一个预训练的模型）
    # model.load_state_dict(torch.load('pretrained_model.pth'))

    # 配置混合精度静态量化
    model = configure_mixed_precision_static_quantization(model)

    # 使用校准数据集收集统计信息
    print("Calibrating the model...")
    calibrate_model(model, train_loader)

    # 转换为真实量化模型
    quantized_model = convert_to_quantized_model(model)

    # 评估量化前的模型性能
    print("Evaluating the model before quantization...")
    evaluate_model(model, test_loader)

    # 评估量化后的模型性能
    print("Evaluating the model after quantization...")
    evaluate_model(quantized_model, test_loader)

    # 保存量化模型
    torch.save(quantized_model.state_dict(), 'static_quantized_model.pth')

    # 加载量化模型并评估
    loaded_model = SimpleCNN()
    loaded_model.load_state_dict(torch.load('static_quantized_model.pth'))
    loaded_model = configure_mixed_precision_static_quantization(loaded_model)
    loaded_model = convert_to_quantized_model(loaded_model)

    print("Evaluating the loaded quantized model...")
    evaluate_model(loaded_model, test_loader)
