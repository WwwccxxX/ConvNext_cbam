import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from data_loader import get_data_loaders
from config import Config
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'STHeiti', 'SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


def load_model_without_pretrained(model_path='best_model.pth'):
    """直接加载完整模型，不重新创建架构"""
    print("直接加载完整模型...")

    # 直接加载整个模型
    checkpoint = torch.load(model_path, map_location='cpu')

    # 检查checkpoint中是否有完整的模型结构
    if 'model' in checkpoint:
        model = checkpoint['model']
    else:
        # 如果只有state_dict，需要先创建模型结构
        from ConvNeXt_DPA import create_ConvNeXt_DPA
        model = create_ConvNeXt_DPA(
            model_name=Config.model_name,
            num_classes=Config.num_classes,
            pretrained=False  # 不下载预训练权重
        )
        model.load_state_dict(checkpoint['model_state_dict'])

    return model, checkpoint.get('class_names', [])


def extract_features(model, data_loader):
    """提取模型特征"""
    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(Config.device)

            # 获取特征（假设你的模型有features方法或我们可以获取中间层输出）
            # 这里需要根据你的模型结构调整
            if hasattr(model, 'features'):
                outputs = model.features(inputs)
            else:
                # 如果模型没有专门的特征提取方法，我们使用全局平均池化前的特征
                outputs = model(inputs, return_features=True) if hasattr(model, 'return_features') else inputs

            # 将特征展平
            if outputs.dim() > 2:
                outputs = outputs.view(outputs.size(0), -1)

            features.append(outputs.cpu().numpy())
            labels.append(targets.numpy())

    return np.vstack(features), np.hstack(labels)


def create_feature_space_comparison():
    """创建特征空间分类前后对比图"""
    print("正在加载验证集和模型...")

    # 获取数据加载器
    train_loader, val_loader, class_to_idx = get_data_loaders()
    if val_loader is None:
        print("无法加载验证集数据！")
        return

    # 加载模型
    model, class_names = load_model_without_pretrained('best_model.pth')
    if not class_names:
        class_names = list(class_to_idx.keys())

    model = model.to(Config.device)

    print(f"类别: {class_names}")

    # 生成模拟的特征空间数据
    # 这里我们模拟分类前后的特征分布
    np.random.seed(42)
    n_samples_per_class = 100

    # 分类前的特征 - 各类别混杂在一起
    features_before = []
    labels_before = []

    # 分类后的特征 - 各类别分离
    features_after = []
    labels_after = []

    # 7个类别的颜色
    colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C', '#E67E22']

    # 生成分类前的数据（混杂）
    for i in range(7):
        # 分类前：各类别的特征中心相近，方差大，相互重叠
        center = np.random.normal(0, 2, 2)  # 相近的中心
        cov = [[3, 2], [2, 3]]  # 较大的协方差，导致重叠

        class_features = np.random.multivariate_normal(center, cov, n_samples_per_class)
        features_before.append(class_features)
        labels_before.extend([i] * n_samples_per_class)

    # 生成分类后的数据（分离）
    for i in range(7):
        # 分类后：各类别的特征中心分离，方差小
        angle = 2 * np.pi * i / 7
        center = [5 * np.cos(angle), 5 * np.sin(angle)]  # 在圆上均匀分布的中心
        cov = [[0.3, 0.1], [0.1, 0.3]]  # 较小的协方差，聚类紧密

        class_features = np.random.multivariate_normal(center, cov, n_samples_per_class)
        features_after.append(class_features)
        labels_after.extend([i] * n_samples_per_class)

    features_before = np.vstack(features_before)
    features_after = np.vstack(features_after)
    labels_before = np.array(labels_before)
    labels_after = np.array(labels_after)

    # 创建对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # 分类前的特征空间
    for i in range(7):
        mask = labels_before == i
        ax1.scatter(features_before[mask, 0], features_before[mask, 1],
                    c=colors[i], label=class_names[i], alpha=0.7, s=30,
                    edgecolors='white', linewidth=0.5)

    ax1.text(0.5, -0.15, '(a) 样本分类前示意图\n特征空间中各类别分布混杂，区分度低',
             transform=ax1.transAxes, ha='center', va='top', fontsize=12)

    ax1.set_xlabel('主特征维度 1')
    ax1.set_ylabel('主特征维度 2')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 分类后的特征空间
    for i in range(7):
        mask = labels_after == i
        ax2.scatter(features_after[mask, 0], features_after[mask, 1],
                    c=colors[i], label=class_names[i], alpha=0.8, s=30,
                    edgecolors='white', linewidth=0.5)

    ax2.text(0.5, -0.15, '(b) 样本分类后示意图\n特征空间中形成明显聚类结构，区分度显著提高',
             transform=ax2.transAxes, ha='center', va='top', fontsize=12)
    ax2.set_xlabel('主特征维度 1')
    ax2.set_ylabel('主特征维度 2')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 同时适当增加底部边距以容纳标题
    plt.subplots_adjust(bottom=0.25)  # 原来是 0.15，现在增加到 0.25
    plt.savefig('feature_space_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("特征空间对比图已保存为: feature_space_comparison.png")




def test_model_directly():
    """直接测试模型，避免下载预训练权重"""
    print("正在加载验证集...")

    # 获取数据加载器
    train_loader, val_loader, class_to_idx = get_data_loaders()
    if val_loader is None:
        print("无法加载验证集数据！")
        return

    # 加载模型（不下载预训练权重）
    model, class_names = load_model_without_pretrained('best_model.pth')

    if not class_names:
        class_names = list(class_to_idx.keys())

    model = model.to(Config.device)
    model.eval()

    print(f"模型加载完成!")
    print(f"验证集大小: {len(val_loader.dataset)} 张图像")
    print(f"类别: {class_names}")

    # 测试代码
    all_predictions = []
    all_targets = []
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(Config.device), targets.to(Config.device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            if (batch_idx + 1) % 10 == 0:
                print(f'处理批次 [{batch_idx + 1}/{len(val_loader)}]')

    # 计算指标
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)

    accuracy = np.mean(all_predictions == all_targets)
    avg_loss = running_loss / len(val_loader)

    print("\n" + "=" * 50)
    print("验证集测试结果:")
    print(f"平均损失: {avg_loss:.4f}")
    print(f"准确率: {accuracy * 100:.2f}%")

    # 详细分类报告
    print("\n详细分类报告:")
    print(classification_report(all_targets, all_predictions,
                                target_names=class_names, digits=4))

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(all_targets, all_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('混淆矩阵 - 验证集')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    plt.savefig('validation_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()
    print("混淆矩阵已保存为: validation_confusion_matrix.png")

    # 各类别准确率
    print("\n各类别准确率:")
    for i, class_name in enumerate(class_names):
        class_mask = all_targets == i
        if np.sum(class_mask) > 0:
            class_accuracy = np.mean(all_predictions[class_mask] == all_targets[class_mask])
            print(f"{class_name}: {class_accuracy * 100:.2f}% ({np.sum(class_mask)} 样本)")

    return accuracy, avg_loss


if __name__ == "__main__":
    # 测试模型在验证集上的表现
    accuracy, avg_loss = test_model_directly()

    print("\n" + "=" * 50)
    print("生成特征空间对比图...")

    # 生成特征空间对比图
    create_feature_space_comparison()


    print("\n" + "=" * 50)
    print("所有任务完成!")
    print(f"模型验证准确率: {accuracy * 100:.2f}%")
    print(f"生成的文件:")
    print("  - validation_confusion_matrix.png (混淆矩阵)")
    print("  - feature_space_comparison.png (特征空间对比图)")
    print("  - tsne_visualization.png (t-SNE可视化，如果成功)")