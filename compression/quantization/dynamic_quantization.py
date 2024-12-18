import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import time

# 1. 定义模型
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 2. 加载预训练模型（假设你已经有一个训练好的浮点模型）
# 如果没有预训练模型，可以使用随机初始化的模型进行测试
float_model = SimpleNet()
# float_model.load_state_dict(torch.load('pretrained_model.pth'))
float_model.eval()

# 3. 应用动态量化
quantized_model = torch.quantization.quantize_dynamic(
    float_model,  # 要量化的模型
    {nn.Linear},  # 需要量化的模块类型
    dtype=torch.qint8  # 量化后的数据类型
)
quantized_model.eval()

# 4. 准备数据集
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 使用 MNIST 数据集作为示例
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = data.DataLoader(test_dataset, batch_size=64, shuffle=False)

# 5. 定义评估函数
def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data = data.view(data.size(0), -1)  # 将输入展平为 784 维向量
            output = model(data)
            _, predicted = torch.max(output, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# 6. 评估量化前后的模型准确率
float_accuracy = evaluate(float_model, test_loader)
print(f"Float Model Accuracy: {float_accuracy:.2f}%")

quantized_accuracy = evaluate(quantized_model, test_loader)
print(f"Quantized Model Accuracy: {quantized_accuracy:.2f}%")

# 7. 比较模型大小
def get_model_size(model, path='temp.pth'):
    torch.save(model.state_dict(), path)
    size = os.path.getsize(path) / 1e6  # 转换为 MB
    os.remove(path)
    return size

float_model_size = get_model_size(float_model)
print(f"Float Model Size: {float_model_size:.2f} MB")

quantized_model_size = get_model_size(quantized_model)
print(f"Quantized Model Size: {quantized_model_size:.2f} MB")

compression_ratio = float_model_size / quantized_model_size
print(f"Compression Ratio: {compression_ratio:.2f}x")

# 8. 比较推理速度
def measure_inference_time(model, data_loader, num_batches=10):
    model.eval()
    times = []
    with torch.no_grad():
        for i, (data, _) in enumerate(data_loader):
            if i >= num_batches:
                break
            data = data.view(data.size(0), -1)  # 将输入展平为 784 维向量
            start_time = time.time()
            _ = model(data)
            end_time = time.time()
            times.append(end_time - start_time)
    avg_time = sum(times) / len(times)
    return avg_time

float_inference_time = measure_inference_time(float_model, test_loader, num_batches=10)
print(f"Float Model Inference Time: {float_inference_time:.4f} seconds per batch")

quantized_inference_time = measure_inference_time(quantized_model, test_loader, num_batches=10)
print(f"Quantized Model Inference Time: {quantized_inference_time:.4f} seconds per batch")

speedup = float_inference_time / quantized_inference_time
print(f"Speedup: {speedup:.2f}x")