import torch
import math
import matplotlib.pyplot as plt
import numpy as np

# ===================== 1. 正弦位置编码 Sinusoidal Position Encoding =====================
def sinusoidal_pe(max_len, d_model):
    """
    标准Transformer正弦位置编码
    :param max_len: 最大序列长度
    :param d_model: 词嵌入维度
    :return: 位置编码矩阵 [max_len, d_model]
    """
    pe = torch.zeros(max_len, d_model)
    pos = torch.arange(0, max_len).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe

# ===================== 2. 二维向量旋转 =====================
def rotate_2d(vec, theta):
    """
    二维平面向量旋转
    :param vec: 二维向量 (x, y)
    :param theta: 旋转角度
    :return: 旋转后向量
    """
    c, s = math.cos(theta), math.sin(theta)
    x, y = vec
    return (x * c - y * s, x * s + y * c)

# ===================== 3. 高维 RoPE 实现 =====================
def rope(x, pos):
    """
    高维旋转位置编码 RoPE
    :param x: 输入向量 [seq_len, d_model]
    :param pos: 位置索引
    :return: 加入位置信息后的旋转向量
    """
    d = x.size(-1)
    half_d = d // 2
    # 计算旋转角频率
    theta = torch.pow(10000.0, -2 * torch.arange(0, half_d) / d).to(x.device)
    pos_theta = pos.unsqueeze(1) * theta
    cos = torch.cos(pos_theta)
    sin = torch.sin(pos_theta)

    # 奇偶维度拆分并执行旋转
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    rx = x1 * cos - x2 * sin
    ry = x1 * sin + x2 * cos

    out = torch.zeros_like(x)
    out[..., 0::2] = rx
    out[..., 1::2] = ry
    return out

# ===================== 4. 绘图 & 数值实验 =====================
def generate_all_plots():
    max_len = 50
    d_model = 64
    device = torch.device("cpu")

    # 图1：正弦位置编码热力图 → 命名：pe_heatmap.png
    pe = sinusoidal_pe(max_len, d_model)
    plt.figure(figsize=(10, 6))
    plt.imshow(pe.numpy(), cmap='viridis', aspect='auto')
    plt.colorbar()
    plt.title("Sinusoidal Position Encoding")
    plt.xlabel("Dimension")
    plt.ylabel("Position")
    plt.tight_layout()
    plt.savefig("pe_heatmap.png", dpi=150)
    plt.close()

    # 图2：E+pos 加法 vs RoPE 旋转对比 → 命名：add_vs_rope.png
    emb = torch.randn(max_len, d_model)
    pe = sinusoidal_pe(max_len, d_model)
    e_add_pos = emb + pe
    emb_rope = rope(emb, torch.arange(max_len))

    plt.figure(figsize=(16, 6))
    plt.suptitle("E+Pos (Add) vs RoPE (Rotate)", fontsize=16)
    plt.subplot(1, 2, 1)
    plt.imshow(e_add_pos.numpy(), cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title("E + Pos")
    plt.xlabel("Dimension")
    plt.ylabel("Position")

    plt.subplot(1, 2, 2)
    plt.imshow(emb_rope.numpy(), cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title("RoPE")
    plt.xlabel("Dimension")
    plt.ylabel("Position")
    plt.tight_layout()
    plt.savefig("add_vs_rope.png", dpi=150)
    plt.close()

    # 图3：高维RoPE特征图 → 命名：rope_feature.png
    plt.figure(figsize=(10, 6))
    plt.imshow(emb_rope.numpy(), cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title("High-Dimensional RoPE Feature")
    plt.xlabel("Dimension")
    plt.ylabel("Position")
    plt.tight_layout()
    plt.savefig("rope_feature.png", dpi=150)
    plt.close()

    # 图4：RoPE相对位置性质验证 → 命名：rope_relative.png
    q = torch.randn(1, d_model)
    k = torch.randn(1, d_model)

    # 相同相对距离 Δ=2，不同绝对位置
    dot1 = (rope(q, torch.tensor([1])) @ rope(k, torch.tensor([3])).T).item()
    dot2 = (rope(q, torch.tensor([5])) @ rope(k, torch.tensor([7])).T).item()
    # 不同相对距离 Δ=5
    dot3 = (rope(q, torch.tensor([0])) @ rope(k, torch.tensor([5])).T).item()

    labels = ["pos(1,3) Δ=2", "pos(5,7) Δ=2", "pos(0,5) Δ=5"]
    values = [dot1, dot2, dot3]
    colors = ["blue", "blue", "orange"]

    plt.figure(figsize=(10, 6))
    plt.bar(labels, values, color=colors)
    plt.title("RoPE Relative Position Verification")
    plt.ylabel("Q-K Dot Product")
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("rope_relative.png", dpi=150)
    plt.close()

    # 输出文件清单
    print("✅ 实验图片生成完成：")
    print("1. pe_heatmap.png    正弦位置编码热力图")
    print("2. add_vs_rope.png   E+Pos 与 RoPE 输入方式对比")
    print("3. rope_feature.png  高维RoPE特征图")
    print("4. rope_relative.png RoPE相对位置数值验证")

# ===================== 主入口 =====================
if __name__ == "__main__":
    generate_all_plots()