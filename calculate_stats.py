import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, datasets
import numpy as np
from isic_data_loader import ISIC2018Downloader
import os
from tqdm import tqdm
from config import Config


def calculate_isic_stats():
    """计算ISIC 2018数据集的真实均值和标准差"""

    # 下载和组织数据（如果还没做）
    downloader = ISIC2018Downloader()
    raw_dir = downloader.raw_dir

    # 检查数据是否存在
    if not os.path.exists(os.path.join(raw_dir, "ISIC2018_Task3_Training_Input")) or \
            not os.path.exists(os.path.join(raw_dir, "ISIC2018_Task3_Training_GroundTruth")):
        print("请先下载ISIC 2018数据集！")
        print("下载地址: https://challenge.isic-archive.com/data/")
        return None, None

    # 组织数据
    train_dir, _ = downloader.organize_data()

    # 创建数据集（只做Resize和ToTensor，不做归一化）
    transform = transforms.Compose([
        transforms.Resize((Config.image_size, Config.image_size)),
        transforms.ToTensor()
    ])

    dataset = datasets.ImageFolder(train_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)

    print("正在计算ISIC 2018数据集的均值和标准差...")

    # 初始化变量
    mean = 0.0
    std = 0.0
    nb_samples = 0

    # 遍历所有批次计算均值
    for data, _ in tqdm(dataloader, desc="计算均值"):
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples

    # 遍历所有批次计算标准差
    var = 0.0
    nb_samples = 0
    for data, _ in tqdm(dataloader, desc="计算标准差"):
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        var += ((data - mean.unsqueeze(1)) ** 2).sum([0, 2])
        nb_samples += batch_samples

    std = torch.sqrt(var / (nb_samples * Config.image_size * Config.image_size))

    mean = mean.numpy()
    std = std.numpy()

    print(f"ISIC 2018数据集统计量:")
    print(f"均值 (mean): [{mean[0]:.3f}, {mean[1]:.3f}, {mean[2]:.3f}]")
    print(f"标准差 (std): [{std[0]:.3f}, {std[1]:.3f}, {std[2]:.3f}]")

    # 保存到配置文件
    config_content = f"""
# ISIC 2018数据集统计量 (自动计算)
isic_mean = [{mean[0]:.6f}, {mean[1]:.6f}, {mean[2]:.6f}]
isic_std = [{std[0]:.6f}, {std[1]:.6f}, {std[2]:.6f}]
"""

    with open('isic_stats.py', 'w') as f:
        f.write(config_content)

    print("统计量已保存到 isic_stats.py")

    return mean, std


if __name__ == "__main__":
    calculate_isic_stats()