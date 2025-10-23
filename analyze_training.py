import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.ticker import MaxNLocator

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'STHeiti', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def analyze_training_info(info_path='training_info.pth'):
    """分析训练信息并可视化"""
    print("正在加载训练信息...")

    # 加载训练信息
    training_info = torch.load(info_path, map_location='cpu')

    print("训练信息内容:")
    for key, value in training_info.items():
        if key in ['train_losses', 'val_losses', 'train_accuracies', 'val_accuracies']:
            print(f"{key}: 长度 {len(value)}")
        else:
            print(f"{key}: {value}")

    # 提取训练历史
    train_losses = training_info.get('train_losses', [])
    val_losses = training_info.get('val_losses', [])
    train_accuracies = training_info.get('train_accuracies', [])
    val_accuracies = training_info.get('val_accuracies', [])
    best_accuracy = training_info.get('best_accuracy', 0)
    model_name = training_info.get('model_name', 'Unknown Model')
    training_time = training_info.get('training_time', 0)
    class_names = training_info.get('class_names', [])

    print(f"\n模型: {model_name}")
    print(f"最佳验证准确率: {best_accuracy:.2f}%")
    print(f"训练时间: {training_time / 60:.2f} 分钟")
    print(f"训练周期: {len(train_losses)}")
    print(f"类别: {class_names}")

    # 创建可视化图表 - 只要前两个图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # 1. 损失曲线
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))



    # 2. 准确率曲线
    ax2.plot(epochs, train_accuracies, 'b-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, val_accuracies, 'r-', label='Val Accuracy', linewidth=2)
    ax2.axhline(y=best_accuracy, color='g', linestyle='--',
                label=f'Best Accuracy: {best_accuracy:.2f}%')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))


    plt.tight_layout()
    # 为底部标题预留空间
    plt.subplots_adjust(bottom=0.2)

    # 在底部添加子图标记
    ax1.text(0.5, -0.15, '(a) ConvNeXt-DPA训练损失曲线', transform=ax1.transAxes,
             ha='center', va='top', fontsize=10, fontweight='bold')

    ax2.text(0.5, -0.15, '(b) ConvNeXt-DPA准确率曲线', transform=ax2.transAxes,
             ha='center', va='top', fontsize=10, fontweight='bold')

    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()

    # 打印关键统计信息
    print("\n关键统计信息:")
    print(f"最终训练准确率: {train_accuracies[-1]:.2f}%" if train_accuracies else "无数据")
    print(f"最终验证准确率: {val_accuracies[-1]:.2f}%" if val_accuracies else "无数据")
    print(f"最佳验证准确率: {best_accuracy:.2f}%")

    if val_accuracies:
        max_epoch = np.argmax(val_accuracies) + 1
        print(f"最佳准确率出现在第 {max_epoch} 个epoch")

    print("训练曲线已保存为: training_curves.png")


if __name__ == "__main__":
    # 分析单个模型的训练信息
    analyze_training_info(
        '/Users/wuchengxun/PycharmProjects/ConvNeXt-CBAM/results/DPA_convnext_tiny_v2/training_info.pth')