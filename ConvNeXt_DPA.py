import torch
import torch.nn as nn
import timm
from cbam import CBAM
import torchvision.models as models

class ConvNeXt_DPABlock(nn.Module):
    """改进的ConvNeXt块，集成CBAM注意力和渐进式大核卷积分解"""

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, kernel_size=7):
        super().__init__()

        # 🎯 创新点1：大核卷积分解
        self.dwconv1 = nn.Conv2d(dim, dim, kernel_size=(1, kernel_size),
                                 padding=(0, kernel_size // 2), groups=dim)
        self.dwconv2 = nn.Conv2d(dim, dim, kernel_size=(kernel_size, 1),
                                 padding=(kernel_size // 2, 0), groups=dim)

        # 原始ConvNeXt组件
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = nn.Identity()

        # 🎯 创新点2：CBAM注意力
        self.cbam = CBAM(dim)

        # 保存核大小信息用于显示
        self.kernel_size = kernel_size

    def forward(self, x):
        input = x

        # 🎯 应用分解的大核卷积
        x = self.dwconv1(x)  # 水平方向卷积 (1xK)
        x = self.dwconv2(x)  # 垂直方向卷积 (Kx1)

        # 后续保持不变
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)

        # 🎯 应用CBAM注意力
        x = self.cbam(x)

        x = input + self.drop_path(x)
        return x


def create_ConvNeXt_DPABlock(model_name='convnext_tiny', num_classes=7, pretrained=True):
    """创建改进的ConvNeXt模型 - 支持预训练权重、CBAM和渐进式大核卷积"""
    try:
        if pretrained:
            print("正在为改进模型加载预训练权重...")
            if model_name == 'convnext_tiny':
                original_model = models.convnext_tiny(weights='IMAGENET1K_V1')
            elif model_name == 'convnext_small':
                original_model = models.convnext_small(weights='IMAGENET1K_V1')
            elif model_name == 'convnext_base':
                original_model = models.convnext_base(weights='IMAGENET1K_V1')
            elif model_name == 'convnext_large':
                original_model = models.convnext_large(weights='IMAGENET1K_V1')
            else:
                raise ValueError(f"不支持的模型: {model_name}")
        else:
            print("改进模型使用随机初始化权重...")
            if model_name == 'convnext_tiny':
                original_model = models.convnext_tiny(weights=None)
            elif model_name == 'convnext_small':
                original_model = models.convnext_small(weights=None)
            elif model_name == 'convnext_base':
                original_model = models.convnext_base(weights=None)
            elif model_name == 'convnext_large':
                original_model = models.convnext_large(weights=None)
            else:
                raise ValueError(f"不支持的模型: {model_name}")

    except Exception as e:
        print(f"改进模型预训练权重加载失败: {e}")
        print("使用随机初始化权重...")
        if model_name == 'convnext_tiny':
            original_model = models.convnext_tiny(weights=None)
        elif model_name == 'convnext_small':
            original_model = models.convnext_small(weights=None)
        elif model_name == 'convnext_base':
            original_model = models.convnext_base(weights=None)
        elif model_name == 'convnext_large':
            original_model = models.convnext_large(weights=None)
        else:
            raise ValueError(f"不支持的模型: {model_name}")

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

    # 🎯 创新点3：渐进式核增长配置
    # Stage1: 小核捕获局部纹理 → Stage4: 大核捕获全局形态
    progressive_kernel_sizes = [7, 11, 13, 13]  # 随着网络加深，核逐渐变大

    print(f"渐进式核增长配置: {progressive_kernel_sizes}")

    # 保存原始权重状态字典
    original_state_dict = original_model.state_dict().copy()

    # 替换原始块为改进块
    stage_idx = 0
    total_blocks_replaced = 0

    for name, module in original_model.named_children():
        if name.startswith('stages'):
            stage = module
            new_stage_blocks = nn.ModuleList()

            # 获取当前阶段的核大小
            current_stage_kernel = progressive_kernel_sizes[stage_idx]
            print(
                f"阶段 {stage_idx + 1}: 使用核大小 {current_stage_kernel}×{current_stage_kernel} (分解为 1×{current_stage_kernel} + {current_stage_kernel}×1)")

            for block_idx_in_stage, block in enumerate(stage):
                block_dim = dims[stage_idx]

                # 创建改进的块，传入渐进式核大小
                improved_block = ConvNeXt_DPABlock(
                    block_dim,
                    kernel_size=current_stage_kernel  # 🎯 传入当前阶段的核大小
                )

                # 权重复制（保持不变）
                improved_block.dwconv.weight.data = block.dwconv.weight.data.clone()
                improved_block.dwconv.bias.data = block.dwconv.bias.data.clone()
                improved_block.norm.weight.data = block.norm.weight.data.clone()
                improved_block.norm.bias.data = block.norm.bias.data.clone()
                improved_block.pwconv1.weight.data = block.pwconv1.weight.data.clone()
                improved_block.pwconv1.bias.data = block.pwconv1.bias.data.clone()
                improved_block.pwconv2.weight.data = block.pwconv2.weight.data.clone()
                improved_block.pwconv2.bias.data = block.pwconv2.bias.data.clone()

                if improved_block.gamma is not None and block.gamma is not None:
                    improved_block.gamma.data = block.gamma.data.clone()

                new_stage_blocks.append(improved_block)
                total_blocks_replaced += 1

            setattr(original_model, name, new_stage_blocks)
            stage_idx += 1

    # 修改分类头
    in_features = original_model.classifier[2].in_features
    original_model.classifier[2] = nn.Linear(in_features, num_classes)

    # 如果使用预训练权重，处理分类头
    if pretrained:
        print("正在初始化改进模型的分类头...")
        current_state_dict = original_model.state_dict()

        for name, param in original_state_dict.items():
            if 'classifier' not in name and name in current_state_dict:
                current_state_dict[name].data.copy_(param.data)

        original_model.load_state_dict(current_state_dict, strict=False)

    # 打印模型信息
    print(f"创建改进模型: {model_name} + CBAM + 渐进式大核卷积分解")
    print(f"渐进式核增长: {progressive_kernel_sizes}")
    print(f"分类头: {in_features} -> {num_classes}")
    print(f"改进块总数: {total_blocks_replaced}")
    print(f"参数总量: {sum(p.numel() for p in original_model.parameters()):,}")

    return original_model