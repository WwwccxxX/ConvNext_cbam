import torch
from torchvision import transforms
import os


class Config:
    # 数据参数
    data_dir = "./ISIC2018"  # 数据将下载到这里
    image_size = 224
    batch_size = 32
    num_workers = 4
    num_classes = 7

    # 训练参数
    epochs = 100
    learning_rate = 3e-4
    weight_decay = 0.01
    warmup_epochs = 5

    # 早停参数
    patience = 15  # 新增：早停耐心值

    # 梯度累积（如果GPU内存不足时使用）
    accumulation_steps = 1  # 设置为1表示不使用，可调整为2或4

    # 标签平滑参数
    label_smoothing = 0.1  # 新增：标签平滑系数

    # 模型选择
    model_name = "convnext_tiny"  # 可选: convnext_tiny, convnext_small, etc.

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 固定随机种子（确保可重复性）
    seed = 42

    # 归一化统计量
    use_imagenet_stats = False
    isic_mean = [0.755, 0.539, 0.568]
    isic_std = [0.143, 0.155, 0.170]


def get_transforms():
    """获取数据预处理变换 - 增强版"""
    if Config.use_imagenet_stats:
        mean, std = Config.imagenet_mean, Config.imagenet_std
    else:
        mean, std = Config.isic_mean, Config.isic_std

    train_transform = transforms.Compose([
        transforms.Resize((Config.image_size, Config.image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        # 增强的颜色抖动
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        # 添加高斯模糊 - 对医学图像有效
        transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 2.0))], p=0.3),
        # 添加随机灰度 - 增加对颜色变化的鲁棒性
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
        # 添加随机擦除 - 类似CutMix的简化版
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((Config.image_size, Config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    return train_transform, val_transform