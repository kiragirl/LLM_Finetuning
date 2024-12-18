import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import prune

# 定义一个简单的全连接神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 创建模型实例
model = SimpleNet()

# 模拟训练数据
dummy_input = torch.randn(64, 784)  # 假设输入是28x28图像展平后的向量
dummy_target = torch.randint(0, 10, (64,))  # 假设输出是10类分类问题

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型（这里仅做演示，实际训练需要更多的epoch和真实数据）
def train_model(model, input_data, target, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    output = model(input_data)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    return loss.item()

# 进行一轮训练
print("Training the model...")
for epoch in range(5):  # 这里只进行5个epoch的训练
    loss = train_model(model, dummy_input, dummy_target, optimizer, criterion)
    print(f"Epoch {epoch+1}, Loss: {loss:.4f}")

# 定义剪枝比例
pruning_amount = 0.2  # 每次剪枝20%的权重

# 对所有线性层应用L1范数非结构化剪枝
def apply_pruning(model, amount):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            prune.l1_unstructured(module, name='weight', amount=amount)
            prune.l1_unstructured(module, name='bias', amount=amount)

# 应用剪枝
print("\nApplying pruning...")
apply_pruning(model, pruning_amount)

# 移除已经剪枝的参数，使模型更紧凑
def remove_pruned_params(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            try:
                prune.remove(module, 'weight')
                prune.remove(module, 'bias')
            except AttributeError:
                pass  # 如果已经移除了，则跳过

# 移除剪枝后的零值参数
remove_pruned_params(model)

# 微调模型
print("\nFine-tuning the pruned model...")
for epoch in range(3):  # 这里只进行3个epoch的微调
    loss = train_model(model, dummy_input, dummy_target, optimizer, criterion)
    print(f"Fine-tune Epoch {epoch+1}, Loss: {loss:.4f}")

# 打印最终模型的参数数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nFinal number of parameters: {count_parameters(model)}")