import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from network import Generator, Discriminator
from dataloader import get_dataloader
from tqdm import tqdm
import multiprocessing

# 超参数
img_dim = 64
lr = 0.0002
epochs = 5
batch_size = 128
G_DIMENSION = 100
beta1 = 0.5
beta2 = 0.999
output_path = 'output'
real_label = 1
fake_label = 0

# 设置设备（优先 MPS > CUDA > CPU）
device = torch.device(
    "mps" if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)
print(f"Using device: {device}")

def main():
    # 数据加载
    train_loader = get_dataloader(
        batch_size=batch_size,
        img_dim=img_dim,
        pin_memory=False if str(device) == "mps" else True  # MPS 不支持 pin_memory
    )

    # 定义模型
    netD = Discriminator().to(device)
    netG = Generator().to(device)

    # 定义损失函数和优化器
    criterion = nn.BCELoss()
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, beta2))

    # 训练过程
    losses = [[], []]
    plt.ioff()
    
    for epoch in range(epochs):
        for batch_id, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False)):
            ############################
            # (1) 更新判别器 D
            ###########################
            netD.zero_grad()
            real_cpu = data.to(device)
            current_batch_size = real_cpu.size(0)
            label = torch.full((current_batch_size,), real_label, dtype=torch.float, device=device)

            output = netD(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()

            noise = torch.randn(current_batch_size, G_DIMENSION, 1, 1, device=device)
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) 更新生成器 G
            ###########################
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake).view(-1)
            errG = criterion(output, label)
            errG.backward()
            optimizerG.step()

            losses[0].append(errD.item())
            losses[1].append(errG.item())

            # 每 100 个 batch 保存一次 loss 曲线
            if batch_id % 100 == 0:
                plt.figure(figsize=(15, 6))
                plt.title('Generator and Discriminator Loss During Training')
                plt.xlabel('Number of Batch')
                plt.plot(np.arange(len(losses[0])), np.array(losses[0]), label='D Loss')
                plt.plot(np.arange(len(losses[1])), np.array(losses[1]), label='G Loss')
                plt.legend()
                plt.savefig(os.path.join(output_path, 'loss_curve_temp.png'))
                plt.close()

    # 训练结束后的操作
    torch.save(netG.state_dict(), "generator.params")
    print("Generator model saved as generator.params")

    # 绘制最终 loss 曲线
    plt.figure(figsize=(15, 6))
    plt.title('Final Generator and Discriminator Loss')
    plt.xlabel('Number of Batch')
    plt.plot(np.arange(len(losses[0])), np.array(losses[0]), label='D Loss')
    plt.plot(np.arange(len(losses[1])), np.array(losses[1]), label='G Loss')
    plt.legend()
    plt.savefig(os.path.join(output_path, 'final_loss_curve.png'))
    plt.close()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()