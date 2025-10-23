from matplotlib import pyplot as plt


def draw_label_smoothing():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # 数据
    categories = ['NV', 'BKL', 'MEL', 'AKIEC', 'VASC']
    hard_labels = [0, 1, 0, 0, 0]  # 真实类别是BKL
    epsilon = 0.1
    smoothed_labels = [epsilon / len(categories) for _ in categories]
    smoothed_labels[1] = 1 - epsilon + epsilon / len(categories)  # 真实类别

    # 左侧：硬标签
    bars1 = ax1.bar(categories, hard_labels, color=['lightcoral', 'lightgreen', 'lightcoral',
                                                    'lightcoral', 'lightcoral'])
    ax1.set_ylim(0, 1.2)
    ax1.set_title('标准交叉熵 - 硬标签', fontsize=14, weight='bold', pad=15)
    ax1.set_ylabel('标签值', fontsize=12, weight='bold')

    # 在柱子上添加数值
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
                     f'{height}', ha='center', va='bottom', fontsize=11, weight='bold')

    # 右侧：平滑标签
    bars2 = ax2.bar(categories, smoothed_labels, color=['lightblue', 'lightgreen', 'lightblue',
                                                        'lightblue', 'lightblue'])
    ax2.set_ylim(0, 1.2)
    ax2.set_title('标签平滑交叉熵 - 软标签', fontsize=14, weight='bold', pad=15)
    ax2.set_ylabel('标签值', fontsize=12, weight='bold')

    # 在柱子上添加数值
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.05,
                 f'{height:.2f}', ha='center', va='bottom', fontsize=11, weight='bold')

    # 添加说明文本
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax1.text(0.05, 0.95, '特点:\n• 硬标签\n• 过度自信风险\n• 易过拟合',
             transform=ax1.transAxes, fontsize=11, verticalalignment='top', bbox=props)

    ax2.text(0.05, 0.95, f'特点 (ε={epsilon}):\n• 软标签\n• 缓解过拟合\n• 提升泛化能力',
             transform=ax2.transAxes, fontsize=11, verticalalignment='top', bbox=props)

    # 整体标题
    fig.suptitle('标签平滑交叉熵损失示意图', fontsize=16, weight='bold', y=0.98)

    plt.tight_layout()
    plt.show()


draw_label_smoothing()