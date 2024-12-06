import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision.models as models

# 加载预训练的 ResNet-18 模型
model = models.resnet18(pretrained=True)

# 打印模型结构
print(model)
print("-------------------------------------------------------------------")
# 打印模型参数
for name, param in model.named_parameters():
    print(f"Layer: {name}, Size: {param.size()}")

for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        print(f"Remaining weights in {name}: {torch.sum(module.weight != 0)}")


# 定义一个函数来对每个卷积层进行局部剪枝
def prune_model_locally(model, pruning_amount=0.2):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # 对每个卷积层应用 L1 剪枝
            prune.l1_unstructured(module, name='weight', amount=pruning_amount)
            print(f"Pruned {name} with {pruning_amount * 100}% of weights")


# 对模型进行局部剪枝，剪掉 20% 的权重
prune_model_locally(model, pruning_amount=0.2)
print("-------------------------------------------------------------------")
# 检查剪枝后的模型
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        print(f"Remaining weights in {name}: {torch.sum(module.weight != 0)}")
