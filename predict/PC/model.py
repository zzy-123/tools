# model.py

import torch
# 直接从torchvision.models.segmentation导入模型和它的权重枚举
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights


def get_model(device: torch.device):
    """
    加载一个预训练的DeepLabV3+模型。

    Args:
        device (torch.device): 模型将要运行的设备 (例如 'cuda' 或 'cpu')。

    Returns:
        torch.nn.Module: 加载并设置为评估模式的PyTorch模型。
    """

    # 1. 指定要使用的预训练权重。
    #    'DEFAULT' 会自动获取当前最推荐的版本。
    weights = DeepLabV3_ResNet101_Weights.DEFAULT

    # 2. 在加载模型时传入weights对象
    model = deeplabv3_resnet101(weights=weights)

    # 将模型移动到指定的设备（GPU或CPU）
    model.to(device)

    # 将模型设置为评估模式
    model.eval()

    print("✅ DeepLabV3+ (ResNet-101) model loaded and set to evaluation mode.")
    return model