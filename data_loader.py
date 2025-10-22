import os
import requests
import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
import shutil
from config import Config, get_transforms


class ISIC2018Downloader:
    """自动下载和处理ISIC 2018数据集的类"""

    def __init__(self, data_dir=Config.data_dir):
        self.data_dir = data_dir
        self.raw_dir = os.path.join(data_dir, "raw")
        self.processed_dir = os.path.join(data_dir, "processed")
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

        # ISIC 2018数据集信息 - 添加验证集
        self.dataset_files = {
            "train_images": "ISIC2018_Task3_Training_Input.zip",
            "train_labels": "ISIC2018_Task3_Training_GroundTruth.zip",
            "val_images": "ISIC2018_Task3_Validation_Input.zip",
            "val_labels": "ISIC2018_Task3_Validation_GroundTruth.zip"
        }

        # 类别名称对应关系
        self.class_names = {
            0: "MEL",  # Melanoma
            1: "NV",  # Melanocytic nevus
            2: "BCC",  # Basal cell carcinoma
            3: "AK",  # Actinic keratosis
            4: "BKL",  # Benign keratosis
            5: "DF",  # Dermatofibroma
            6: "VASC"  # Vascular lesion
        }

    def organize_training_data(self):
        """组织训练数据为ImageFolder格式"""
        print("正在组织训练数据...")

        # 创建训练目录结构
        train_dir = os.path.join(self.processed_dir, "train")
        for class_name in self.class_names.values():
            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)

        # 读取训练标签文件
        labels_path = os.path.join(self.raw_dir,
                                   "ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv")
        df = pd.read_csv(labels_path)

        # 训练图像目录
        images_dir = os.path.join(self.raw_dir, "ISIC2018_Task3_Training_Input")

        # 复制训练图像到对应目录
        self._copy_images(df, images_dir, train_dir, "训练集")

        print("训练数据组织完成!")
        return train_dir

    def organize_validation_data(self):
        """组织官方验证数据为ImageFolder格式"""
        print("正在组织官方验证数据...")

        # 创建验证目录结构
        val_dir = os.path.join(self.processed_dir, "val")
        for class_name in self.class_names.values():
            os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

        # 读取验证标签文件
        labels_path = os.path.join(self.raw_dir,
                                   "ISIC2018_Task3_Validation_GroundTruth/ISIC2018_Task3_Validation_GroundTruth.csv")
        df = pd.read_csv(labels_path)

        # 验证图像目录
        images_dir = os.path.join(self.raw_dir, "ISIC2018_Task3_Validation_Input")

        # 复制验证图像到对应目录
        self._copy_images(df, images_dir, val_dir, "官方验证集")

        print("验证数据组织完成!")
        return val_dir

    def _copy_images(self, df, src_dir, dst_dir, dataset_name):
        """复制图像到目标目录"""
        print(f"正在处理{dataset_name}...")
        copied_count = 0

        for _, row in df.iterrows():
            image_id = row['image']
            # 找到对应的类别 (从MEL, NV, BCC, AKIEC, BKL, DF, VASC列中找到值为1的)
            class_columns = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']
            for i, col in enumerate(class_columns):
                if col in row and row[col] == 1:
                    class_name = self.class_names[i]
                    break
            else:
                # 如果没找到，尝试其他列名格式
                class_idx = None
                for col in row.index:
                    if col != 'image' and row[col] == 1:
                        class_idx = int(col.split('_')[-1]) if 'AKIEC' not in col else 3
                        break

                if class_idx is not None:
                    class_name = self.class_names[class_idx]
                else:
                    print(f"警告: 无法找到图像 {image_id} 的类别")
                    continue

            src_path = os.path.join(src_dir, f"{image_id}.jpg")
            dst_path = os.path.join(dst_dir, class_name, f"{image_id}.jpg")

            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)
                copied_count += 1
            else:
                print(f"警告: 图像文件不存在 {src_path}")

        print(f"{dataset_name}处理完成: {copied_count}/{len(df)} 张图像")


def get_data_loaders():
    """获取数据加载器 - 使用官方验证集"""
    downloader = ISIC2018Downloader()

    # 检查所有必要文件是否存在
    required_paths = [
        os.path.join(downloader.raw_dir, "ISIC2018_Task3_Training_Input"),
        os.path.join(downloader.raw_dir, "ISIC2018_Task3_Training_GroundTruth"),
        os.path.join(downloader.raw_dir, "ISIC2018_Task3_Validation_Input"),
        os.path.join(downloader.raw_dir, "ISIC2018_Task3_Validation_GroundTruth")
    ]

    missing_paths = [path for path in required_paths if not os.path.exists(path)]

    if missing_paths:
        print("请下载完整的ISIC 2018 Task3数据集！")
        print("需要下载的文件:")
        print("1. ISIC2018_Task3_Training_Input.zip")
        print("2. ISIC2018_Task3_Training_GroundTruth.zip")
        print("3. ISIC2018_Task3_Validation_Input.zip")
        print("4. ISIC2018_Task3_Validation_GroundTruth.zip")
        print("缺失的文件/目录:")
        for path in missing_paths:
            print(f"  - {os.path.basename(path)}")
        return None, None, None

    # 组织训练数据
    train_dir = downloader.organize_training_data()

    # 组织官方验证数据
    val_dir = downloader.organize_validation_data()

    # 获取数据变换
    train_transform, val_transform = get_transforms()

    # 创建数据集
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers
    )

    print(f"训练集: {len(train_dataset)} 张图像")
    print(f"官方验证集: {len(val_dataset)} 张图像")
    print(f"类别: {train_dataset.classes}")

    return train_loader, val_loader, train_dataset.class_to_idx