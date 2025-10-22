import torch
import torch.nn as nn
import timm
from cbam import CBAM
import torchvision.models as models


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
    """创建基线ConvNeXt模型 - 使用torchvision的预训练权重"""
    try:
        if pretrained:
            print("正在加载PyTorch官方预训练权重...")
            # 使用torchvision的ConvNeXt，它有更好的下载稳定性
            if model_name == 'convnext_tiny':
                model = models.convnext_tiny(weights='IMAGENET1K_V1')
            elif model_name == 'convnext_small':
                model = models.convnext_small(weights='IMAGENET1K_V1')
            elif model_name == 'convnext_base':
                model = models.convnext_base(weights='IMAGENET1K_V1')
            elif model_name == 'convnext_large':
                model = models.convnext_large(weights='IMAGENET1K_V1')
            else:
                raise ValueError(f"不支持的模型: {model_name}")

            # 修改分类头
            in_features = model.classifier[2].in_features
            model.classifier[2] = torch.nn.Linear(in_features, num_classes)
            print("PyTorch官方预训练权重加载成功!")

        else:
            # 不使用预训练权重
            print("使用随机初始化权重...")
            if model_name == 'convnext_tiny':
                model = models.convnext_tiny(weights=None)
            elif model_name == 'convnext_small':
                model = models.convnext_small(weights=None)
            elif model_name == 'convnext_base':
                model = models.convnext_base(weights=None)
            elif model_name == 'convnext_large':
                model = models.convnext_large(weights=None)
            else:
                raise ValueError(f"不支持的模型: {model_name}")

            # 修改分类头
            in_features = model.classifier[2].in_features
            model.classifier[2] = torch.nn.Linear(in_features, num_classes)

    except Exception as e:
        print(f"预训练权重加载失败: {e}")
        print("使用随机初始化权重...")
        # 回退到随机初始化
        if model_name == 'convnext_tiny':
            model = models.convnext_tiny(weights=None)
        elif model_name == 'convnext_small':
            model = models.convnext_small(weights=None)
        elif model_name == 'convnext_base':
            model = models.convnext_base(weights=None)
        elif model_name == 'convnext_large':
            model = models.convnext_large(weights=None)
        else:
            raise ValueError(f"不支持的模型: {model_name}")

        # 修改分类头
        in_features = model.classifier[2].in_features
        model.classifier[2] = torch.nn.Linear(in_features, num_classes)

    # 打印模型信息
    print(f"创建模型: {model_name}")
    print(f"分类头: {in_features} -> {num_classes}")
    print(f"参数总量: {sum(p.numel() for p in model.parameters()):,}")

    return model