import torch
import torch.nn as nn
import timm
from cbam import CBAM


class ImprovedConvNeXtBlock(nn.Module):
    """改进的ConvNeXt块，集成CBAM注意力"""

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        # 原始ConvNeXt组件
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = nn.Identity()  # 简化，不使用drop path

        # 你的创新：添加CBAM注意力
        self.cbam = CBAM(dim)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        # 你的创新在这里：应用CBAM注意力
        x = self.cbam(x)

        x = input + self.drop_path(x)
        return x


def create_improved_convnext(model_name='convnext_tiny', num_classes=7, pretrained=True):
    """创建改进的ConvNeXt模型"""
    # 加载原始模型
    original_model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)

    # 根据模型名称确定配置
    if 'tiny' in model_name:
        depths = [3, 3, 9, 3]
        dims = [96, 192, 384, 768]
    elif 'small' in model_name:
        depths = [3, 3, 27, 3]
        dims = [96, 192, 384, 768]
    elif 'base' in model_name:
        depths = [3, 3, 27, 3]
        dims = [128, 256, 512, 1024]
    elif 'large' in model_name:
        depths = [3, 3, 27, 3]
        dims = [192, 384, 768, 1536]
    else:
        raise ValueError(f"不支持的模型: {model_name}")

    # 替换原始块为改进块
    stage_idx = 0
    block_idx = 0

    # 遍历所有模块
    for name, module in original_model.named_children():
        if name.startswith('stages'):
            # 这是卷积阶段
            stage = module
            new_stage_blocks = nn.ModuleList()

            for block in stage:
                # 获取块的维度
                block_dim = dims[stage_idx]
                # 创建改进的块
                improved_block = ImprovedConvNeXtBlock(block_dim)
                new_stage_blocks.append(improved_block)
                block_idx += 1

            # 替换原始阶段
            setattr(original_model, name, new_stage_blocks)
            stage_idx += 1
            block_idx = 0

    return original_model


def create_baseline_convnext(model_name='convnext_tiny', num_classes=7, pretrained=True):
    """创建基线ConvNeXt模型"""
    return timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)