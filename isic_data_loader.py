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

        # ISIC 2018数据集信息
        self.dataset_files = {
            "images": "ISIC2018_Task3_Training_Input.zip",
            "labels": "ISIC2018_Task3_Training_GroundTruth.zip"
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

    def download_file(self, url, filename):
        """下载文件"""
        filepath = os.path.join(self.raw_dir, filename)

        if os.path.exists(filepath):
            print(f"文件已存在: {filename}")
            return filepath

        print(f"正在下载 {filename}...")
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(filepath, 'wb') as file:
            for data in response.iter_content(chunk_size=1024):
                file.write(data)

        print(f"下载完成: {filename}")
        return filepath

    def extract_zip(self, zip_path, extract_to):
        """解压zip文件"""
        print(f"正在解压 {os.path.basename(zip_path)}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("解压完成")

    def organize_data(self):
        """组织数据为ImageFolder格式"""
        print("正在组织数据...")

        # 创建训练和验证目录结构
        train_dir = os.path.join(self.processed_dir, "train")
        val_dir = os.path.join(self.processed_dir, "val")

        for class_name in self.class_names.values():
            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

        # 读取标签文件
        labels_path = os.path.join(self.raw_dir,
                                   "ISIC2018_Task3_Training_GroundTruth/ISIC2018_Task3_Training_GroundTruth.csv")
        df = pd.read_csv(labels_path)

        # 图像目录
        images_dir = os.path.join(self.raw_dir, "ISIC2018_Task3_Training_Input")

        # 分割训练集和验证集
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=Config.seed,
                                            stratify=df.iloc[:, 1:].idxmax(axis=1))

        # 复制图像到对应目录
        self._copy_images(train_df, images_dir, train_dir, "训练集")
        self._copy_images(val_df, images_dir, val_dir, "验证集")

        print("数据组织完成!")
        return train_dir, val_dir

    def _copy_images(self, df, src_dir, dst_dir, dataset_name):
        """复制图像到目标目录"""
        print(f"正在处理{dataset_name}...")
        for _, row in df.iterrows():
            image_id = row['image']
            # 找到对应的类别
            class_idx = row.iloc[1:].idxmax()
            class_name = self.class_names[int(class_idx.split('_')[-1])]

            src_path = os.path.join(src_dir, f"{image_id}.jpg")
            dst_path = os.path.join(dst_dir, class_name, f"{image_id}.jpg")

            if os.path.exists(src_path):
                shutil.copy2(src_path, dst_path)

        print(f"{dataset_name}处理完成: {len(df)} 张图像")


def get_data_loaders():
    """获取数据加载器"""
    # 下载和组织数据
    downloader = ISIC2018Downloader()

    # 注意: 由于版权原因，这里无法自动下载ISIC数据集
    # 你需要手动从 https://challenge.isic-archive.com/data/ 下载以下文件:
    # 1. ISIC2018_Task3_Training_Input.zip
    # 2. ISIC2018_Task3_Training_GroundTruth.zip
    # 然后放在 ./ISIC2018/raw/ 目录下

    raw_dir = downloader.raw_dir
    if not os.path.exists(os.path.join(raw_dir, "ISIC2018_Task3_Training_Input")) or \
            not os.path.exists(os.path.join(raw_dir, "ISIC2018_Task3_Training_GroundTruth")):
        print("请手动下载ISIC 2018数据集并放在 ./ISIC2018/raw/ 目录下")
        print("下载地址: https://challenge.isic-archive.com/data/")
        print("需要下载的文件:")
        print("1. ISIC2018_Task3_Training_Input.zip")
        print("2. ISIC2018_Task3_Training_GroundTruth.zip")
        return None, None

    # 组织数据
    train_dir, val_dir = downloader.organize_data()

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
    print(f"验证集: {len(val_dataset)} 张图像")
    print(f"类别: {train_dataset.classes}")

    return train_loader, val_loader, train_dataset.class_to_idx