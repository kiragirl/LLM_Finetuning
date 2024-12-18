import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torchvision.models as models

# 加载预训练的 ResNet-18 模型
model = models.resnet18(pretrained=True)


for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        print(f"Remaining weights in {name}: {torch.sum(module.weight != 0)}")


# 定义一个函数来获取所有可剪枝的参数
def get_all_prunable_params(model):
    params_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            params_to_prune.append((module, 'weight'))
    return params_to_prune


# 获取所有可剪枝的参数
params_to_prune = get_all_prunable_params(model)

# 对所有可剪枝的参数进行全局剪枝，剪掉 20% 的权重
prune.global_unstructured(
    params_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2,
)
print("-------------------------------------------------------------------")
# 检查剪枝后的模型
for name, module in model.named_modules():
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        print(f"Remaining weights in {name}: {torch.sum(module.weight != 0)}")
