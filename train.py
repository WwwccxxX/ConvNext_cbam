import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from tqdm import tqdm

from config import Config
from data_loader import get_data_loaders
from convnext_cbam import create_baseline_convnext, create_improved_convnext
import torch.nn.functional as F
from ConvNeXt_DPA import create_ConvNeXt_DPABlock
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

TRAIN_VERSION = "v1"  # 每次训练前手动修改这个数字


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, x, target):
        log_probs = F.log_softmax(x, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = (1 - self.smoothing) * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class Trainer:
    def __init__(self, model, model_name, train_loader, val_loader, class_names):
        self.model = model
        self.model_name = model_name
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.class_names = class_names
        self.device = Config.device

        # 损失函数和优化器
        self.criterion = LabelSmoothingCrossEntropy(smoothing=Config.label_smoothing)
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=Config.learning_rate,
            weight_decay=Config.weight_decay
        )

        self.scheduler = CosineAnnealingWarmRestarts(  # 这是学习率调度器
            self.optimizer,
            T_0=10,  # 第一次重启的周期
            T_mult=2,  # 每次重启周期翻倍
            eta_min=1e-6  # 最小学习率
        )

        # 训练记录
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        self.best_accuracy = 0.0

        # 创建保存目录
        self.save_dir = f"results/{model_name}"
        os.makedirs(self.save_dir, exist_ok=True)

        print(f"训练设备: {self.device}")
        print(f"模型: {model_name}")
        print(f"参数总量: {sum(p.numel() for p in model.parameters()):,}")

    def train_epoch(self, epoch):
        """训练一个epoch - 带梯度累积"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # 清零梯度
        self.optimizer.zero_grad()

        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{Config.epochs} [训练]')

        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            # 梯度累积：损失除以累积步数
            loss = loss / Config.accumulation_steps
            loss.backward()

            # 只有达到累积步数时才更新权重
            if (batch_idx + 1) % Config.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                # 更新学习率
                self.scheduler.step(epoch + batch_idx / len(self.train_loader))

            running_loss += loss.item() * Config.accumulation_steps
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item() * Config.accumulation_steps:.4f}',
                'Acc': f'{100. * correct / total:.2f}%',
                'LR': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })

        # 处理最后一个不完整的累积批次
        if (batch_idx + 1) % Config.accumulation_steps != 0:
            self.optimizer.step()
            self.optimizer.zero_grad()

        epoch_loss = running_loss / len(self.train_loader)
        epoch_accuracy = 100. * correct / total

        self.train_losses.append(epoch_loss)
        self.train_accuracies.append(epoch_accuracy)

        return epoch_loss, epoch_accuracy

    def validate(self, epoch):
        """验证模型 - 带早停检查"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch + 1}/{Config.epochs} [验证]')
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{100. * correct / total:.2f}%'
                })

        epoch_loss = running_loss / len(self.val_loader)
        epoch_accuracy = 100. * correct / total

        self.val_losses.append(epoch_loss)
        self.val_accuracies.append(epoch_accuracy)

        # 早停检查
        if epoch_accuracy > self.best_accuracy:
            self.best_accuracy = epoch_accuracy
            self.patience_counter = 0  # 重置耐心计数器

            # 修复：创建一个可序列化的配置字典
            config_dict = {
                key: value for key, value in Config.__dict__.items()
                if not key.startswith('_') and not callable(value)
            }

            # 保存最佳模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'accuracy': epoch_accuracy,
                'loss': epoch_loss,
                'class_names': self.class_names,  # 添加类别名称
                'config': config_dict  # 使用可序列化的配置字典
            }, os.path.join(self.save_dir, 'best_model.pth'))
            print(f"新的最佳准确率: {epoch_accuracy:.2f}%")
        else:
            self.patience_counter += 1
            print(f"早停计数: {self.patience_counter}/{Config.patience}")

        return epoch_loss, epoch_accuracy

    def plot_training_history(self):
        """绘制训练历史"""
        plt.figure(figsize=(12, 4))

        # 损失曲线
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{self.model_name} - Loss Curve')
        plt.legend()
        plt.grid(True)

        # 准确率曲线
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, label='Train Accuracy')
        plt.plot(self.val_accuracies, label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title(f'{self.model_name} - Accuracy Curve')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
        plt.show()

        print(f"训练历史图已保存到: {os.path.join(self.save_dir, 'training_history.png')}")

    def train(self):
        """完整训练流程"""
        print(f"开始训练 {self.model_name}...")
        start_time = time.time()

        # 初始化早停变量
        self.patience_counter = 0
        self.best_accuracy = 0.0

        for epoch in range(Config.epochs):
            if self.patience_counter >= Config.patience:
                print(f"早停触发！在 epoch {epoch + 1} 停止训练")
                break

            # 训练
            train_loss, train_acc = self.train_epoch(epoch)

            # 验证
            val_loss, val_acc = self.validate(epoch)

            # 更新学习率
            self.scheduler.step()

            # 打印epoch总结
            print(f'Epoch {epoch + 1}/{Config.epochs}: '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, '
                  f'Best Val Acc: {self.best_accuracy:.2f}%')

        # 绘制训练历史
        self.plot_training_history()

        training_time = time.time() - start_time
        print(f"训练完成! 最佳验证准确率: {self.best_accuracy:.2f}%")
        print(f"总训练时间: {training_time / 60:.2f} 分钟")

        # 保存最终训练记录
        training_info = {
            'model_name': self.model_name,
            'best_accuracy': self.best_accuracy,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'training_time': training_time,
            'class_names': self.class_names  # 保存类别名称
        }

        torch.save(training_info, os.path.join(self.save_dir, 'training_info.pth'))

        # 修复：创建一个可序列化的配置字典
        config_dict = {
            key: value for key, value in Config.__dict__.items()
            if not key.startswith('_') and not callable(value)
        }

        # 保存最终模型
        torch.save({
            'epoch': Config.epochs,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'accuracy': self.best_accuracy,
            'class_names': self.class_names,  # 保存类别名称
            'config': config_dict  # 使用可序列化的配置字典
        }, os.path.join(self.save_dir, 'final_model.pth'))

        print(f"最终模型已保存到: {os.path.join(self.save_dir, 'final_model.pth')}")

        return self.best_accuracy


def main():
    """主函数"""
    # 设置随机种子
    torch.manual_seed(Config.seed)
    np.random.seed(Config.seed)

    # 获取数据
    print("正在加载数据...")
    train_loader, val_loader, class_to_idx = get_data_loaders()
    if train_loader is None:
        return

    class_names = list(class_to_idx.keys())
    print(f"类别: {class_names}")

    # # 训练基线模型
    # print("\n" + "=" * 50)
    # print("训练基线ConvNeXt模型...")
    # baseline_model = create_baseline_convnext(
    #     model_name=Config.model_name,
    #     num_classes=Config.num_classes
    # )
    # baseline_model = baseline_model.to(Config.device)
    #
    # baseline_trainer = Trainer(
    #     baseline_model,
    #     f"baseline_{Config.model_name}_{TRAIN_VERSION}",
    #     train_loader,
    #     val_loader,
    #     class_names
    # )
    # baseline_accuracy = baseline_trainer.train()
    #
    # # 修复：创建一个可序列化的配置字典
    # config_dict = {
    #     key: value for key, value in Config.__dict__.items()
    #     if not key.startswith('_') and not callable(value)
    # }
    #
    # # 保存完整的基线模型用于推理
    # torch.save({
    #     'model_state_dict': baseline_model.state_dict(),
    #     'class_names': class_names,
    #     'class_to_idx': class_to_idx,
    #     'accuracy': baseline_accuracy,
    #     'config': config_dict  # 使用可序列化的配置字典
    # }, 'baseline_model_complete.pth')
    # print("基线模型已保存为: baseline_model_complete.pth")

    # 训练ConvNeXt_DPA模型
    print("\n" + "=" * 50)
    print("训练ConvNeXt_DPABlock模型...")
    dpa_model = create_ConvNeXt_DPABlock(
        model_name=Config.model_name,
        num_classes=Config.num_classes
    )
    dpa_model = dpa_model.to(Config.device)

    dpa_trainer = Trainer(
        dpa_model,
        f"DPA_{Config.model_name}_{TRAIN_VERSION}",
        train_loader,
        val_loader,
        class_names
    )
    dpa_accuracy = dpa_trainer.train()

    config_dict = {
        key: value for key, value in Config.__dict__.items()
        if not key.startswith('_') and not callable(value)
    }

    # 保存完整的DPA模型用于推理
    torch.save({
        'model_state_dict': dpa_model.state_dict(),
        'class_names': class_names,
        'class_to_idx': class_to_idx,
        'accuracy': dpa_accuracy,
        'config': config_dict  # 使用可序列化的配置字典
    }, 'ConvNeXt_DPABlock_model_complete.pth')
    print("ConvNeXt_DPABlock模型已保存为: ConvNeXt_DPABlock_model_complete.pth")



    # 训练改进模型（注释掉，先跑基线）

    # print("\n" + "="*50)
    # print("训练改进的ConvNeXt模型...")
    # improved_model = create_improved_convnext(
    #     model_name=Config.model_name,
    #     num_classes=Config.num_classes
    # )
    # improved_model = improved_model.to(Config.device)

    # improved_trainer = Trainer(
    #     improved_model,
    #     f"improved_{Config.model_name}_{TRAIN_VERSION}",
    #     train_loader,
    #     val_loader,
    #     class_names
    # )
    # improved_accuracy = improved_trainer.train()

    # config_dict = {
    #     key: value for key, value in Config.__dict__.items()
    #     if not key.startswith('_') and not callable(value)
    # }

    # # 保存完整的改进模型用于推理
    # torch.save({
    #     'model_state_dict': improved_model.state_dict(),
    #     'class_names': class_names,
    #     'class_to_idx': class_to_idx,
    #     'accuracy': improved_accuracy,
    #     'config': config_dict  # 使用可序列化的配置字典
    # }, 'improved_model_complete.pth')
    # print("改进模型已保存为: improved_model_complete.pth")

    # # 对比结果
    # print("\n" + "="*50)
    # print("模型对比结果:")
    # print(f"基线模型 ({Config.model_name}): {baseline_accuracy:.2f}%")
    # print(f"改进模型 ({Config.model_name} + CBAM): {improved_accuracy:.2f}%")
    # print(f"性能提升: {improved_accuracy - baseline_accuracy:.2f}%")


if __name__ == "__main__":
    main()