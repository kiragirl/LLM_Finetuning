import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision.models as models
import matplotlib.pyplot as plt
# 加载预训练的 ResNet-18 模型
model = models.resnet18(pretrained=True)


def count_nonzero_parameters(model):
    return sum(p.nonzero().size(0) for p in model.parameters() if p.requires_grad)


# 可视化剪枝前的某一层权重分布
print(f"Number of non-zero parameters before pruning: {count_nonzero_parameters(model)}")


# 定义一个函数来对每个卷积层进行结构化剪枝
def prune_model_structured(model, pruning_amount=0.2):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # 对每个卷积层应用结构化剪枝，剪掉 20% 的通道
            prune.ln_structured(module, name='weight', amount=pruning_amount, n=2, dim=0)
            print(f"Pruned {name} with {pruning_amount * 100}% of channels")


# 对模型进行结构化剪枝，剪掉 20% 的通道
prune_model_structured(model, pruning_amount=0.2)

# 检查剪枝后的模型
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        print(f"Remaining channels in {name}: {module.weight.shape[0]}")



print("-----------------")
# 可视化剪枝后的同一层权重分布
print(f"Number of non-zero parameters after pruning: {count_nonzero_parameters(model)}")
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) and hasattr(module, 'weight_mask'):
        print(f"Mask in {name}: {module.weight_mask.sum() / module.weight_mask.numel():.2f} (Ratio of non-zero elements)")