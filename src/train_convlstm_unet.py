"""
ConvLSTM-UNet 时空预测训练脚本（连续值预测）

本脚本使用 `data/interim/npz` 目录下的 npz 数据，实现整幅图的连续值时空预测
（给定过去若干年，预测下一年整幅图的连续值）。
"""
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from scipy import stats
import random

# 设置 matplotlib 字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号


# ==================== 模型定义 ====================

class ConvLSTMCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(input_channels + hidden_channels,
                              4 * hidden_channels,
                              kernel_size=kernel_size,
                              padding=padding)

    def forward(self, x, h, c):
        # x: [B, C_in, H, W]
        # h, c: [B, C_h, H, W]
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        i, f, o, g = torch.chunk(gates, 4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        self.cell = ConvLSTMCell(input_channels, hidden_channels, kernel_size)

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        h = torch.zeros(B, self.cell.hidden_channels, H, W, device=x.device)
        c = torch.zeros_like(h)
        for t in range(T):
            h, c = self.cell(x[:, t], h, c)
        return h  # 只返回最后一个时间步的隐藏状态


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ConvLSTMUNet(nn.Module):
    """简单的 2 层编码 + ConvLSTM bottleneck + 2 层解码的 UNet 结构（连续值输出）。"""
    def __init__(self):
        super().__init__()
        # encoder
        self.enc1 = DoubleConv(1, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        # ConvLSTM bottleneck（在最深层做时间建模）
        self.conv_lstm = ConvLSTM(input_channels=64, hidden_channels=64, kernel_size=3)

        # decoder
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(32 + 64, 32)
        self.up2 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(16 + 1, 16)  # 与原始分辨率下的 1 通道 skip 连接

        # 输出层：连续值（1 个通道）
        self.out_conv = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        # x: [B, T, 1, H, W]
        B, T, C, H, W = x.shape

        # 逐时间步编码，记录最后一年的 skip
        enc1_list = []
        enc2_list = []
        for t in range(T):
            x_t = x[:, t]  # [B, 1, H, W]
            e1 = self.enc1(x_t)          # [B, 32, H, W]
            p1 = self.pool1(e1)          # [B, 32, H/2, W/2]
            e2 = self.enc2(p1)           # [B, 64, H/2, W/2]
            p2 = self.pool2(e2)          # [B, 64, H/4, W/4]
            enc1_list.append(e1)
            enc2_list.append(e2)

        # ConvLSTM 只在最深层输入，每个时间步的 p2
        deep_seq = torch.stack([self.pool2(e2) for e2 in enc2_list], dim=1)  # [B, T, 64, H/4, W/4]
        bottleneck = self.conv_lstm(deep_seq)  # [B, 64, H/4, W/4]

        # 解码使用最后一年的 skip 连接
        e1_last = enc1_list[-1]
        e2_last = enc2_list[-1]

        x_up1 = self.up1(bottleneck)          # [B, 32, H/2, W/2]
        # 调整 x_up1 以匹配 e2_last 的尺寸（处理维度不匹配问题）
        _, _, h_up1, w_up1 = x_up1.shape
        _, _, h_e2, w_e2 = e2_last.shape
        if h_up1 != h_e2 or w_up1 != w_e2:
            # 如果 x_up1 更大，则裁剪（从中心裁剪）
            if h_up1 > h_e2 or w_up1 > w_e2:
                diff_h = h_up1 - h_e2
                diff_w = w_up1 - w_e2
                start_h = diff_h // 2
                end_h = h_up1 - (diff_h - diff_h // 2)
                start_w = diff_w // 2
                end_w = w_up1 - (diff_w - diff_w // 2)
                x_up1 = x_up1[:, :, start_h:end_h, start_w:end_w]
            # 如果 x_up1 更小，则填充（使用零填充）
            else:
                diff_h = h_e2 - h_up1
                diff_w = w_e2 - w_up1
                pad_h = (diff_h // 2, diff_h - diff_h // 2)
                pad_w = (diff_w // 2, diff_w - diff_w // 2)
                x_up1 = F.pad(x_up1, pad=(pad_w[0], pad_w[1], pad_h[0], pad_h[1]), mode='constant', value=0)
        x_cat1 = torch.cat([x_up1, e2_last], dim=1)  # [B, 32+64, H/2, W/2]
        x_dec1 = self.dec1(x_cat1)            # [B, 32, H/2, W/2]

        x_up2 = self.up2(x_dec1)              # [B, 16, H, W]
        # 把原图分辨率下的最后一年原始输入作为最浅层 skip
        x_last = x[:, -1, 0:1]                # [B, 1, H, W]
        # 调整 x_up2 以匹配 x_last 的尺寸（处理维度不匹配问题）
        _, _, h_up2, w_up2 = x_up2.shape
        _, _, h_last, w_last = x_last.shape
        if h_up2 != h_last or w_up2 != w_last:
            # 如果 x_up2 更大，则裁剪（从中心裁剪）
            if h_up2 > h_last or w_up2 > w_last:
                diff_h = h_up2 - h_last
                diff_w = w_up2 - w_last
                start_h = diff_h // 2
                end_h = h_up2 - (diff_h - diff_h // 2)
                start_w = diff_w // 2
                end_w = w_up2 - (diff_w - diff_w // 2)
                x_up2 = x_up2[:, :, start_h:end_h, start_w:end_w]
            # 如果 x_up2 更小，则填充（使用零填充）
            else:
                diff_h = h_last - h_up2
                diff_w = w_last - w_up2
                pad_h = (diff_h // 2, diff_h - diff_h // 2)
                pad_w = (diff_w // 2, diff_w - diff_w // 2)
                x_up2 = F.pad(x_up2, pad=(pad_w[0], pad_w[1], pad_h[0], pad_h[1]), mode='constant', value=0)
        x_cat2 = torch.cat([x_up2, x_last], dim=1)  # [B, 16+1, H, W]
        x_dec2 = self.dec2(x_cat2)            # [B, 16, H, W]

        output = self.out_conv(x_dec2)        # [B, 1, H, W]
        return output.squeeze(1)              # [B, H, W]


# ==================== 数据集定义 ====================

class SpatioTemporalPatchDataset(Dataset):
    """从整幅时序图中随机采样 [window_size, patch, patch] → [patch, patch] 的训练样本（连续值）。"""
    def __init__(self, data, mask, window_size=5, patch_size=128, num_samples=2000, seed=42):
        super().__init__()
        self.data = data  # [T, H, W], float
        self.mask = mask  # [H, W], long, 0=无效
        self.window_size = window_size
        self.patch_size = patch_size
        self.num_samples = num_samples
        self.T, self.H, self.W = data.shape
        self.rng = random.Random(seed)

        # 允许作为 target 年份的范围：[window_size, T-1]
        self.valid_years = list(range(window_size, self.T))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 随机选一个目标年份
        t = self.rng.choice(self.valid_years)

        # 随机选一个中心坐标，保证 patch 完全落在图内
        p = self.patch_size
        i = self.rng.randint(0, self.H - p)
        j = self.rng.randint(0, self.W - p)

        # 如果空值占比>0.3，就重抽
        while True:
            i = self.rng.randint(0, self.H - p)
            j = self.rng.randint(0, self.W - p)
            patch_mask = self.mask[i:i+p, j:j+p]
            # 计算 patch_mask 中值为 1 的元素的数量
            num_ones = (patch_mask == 1).sum()
            # 计算 patch_mask 的总元素数量
            total_elements = self.patch_size * self.patch_size

            # if (num_ones/total_elements < 0.3):
            if (num_ones == 0):
                break

        # 输入：过去 window_size 年的 patch 序列
        x = self.data[t-self.window_size:t, i:i+p, j:j+p]  # [T_w, p, p]
        # 目标：当前年份的连续值图（mask=1 位置设为 NaN）
        y = self.data[t, i:i+p, j:j+p]  # [p, p]

        # 添加 channel 维度，方便 ConvLSTM-UNet 使用： [T_w, 1, p, p]
        x = x.unsqueeze(1)
        return x, y


# ==================== 损失函数 ====================

def masked_mse_loss(pred, target):
    """计算 MSE 损失，忽略 NaN 值"""
    valid_mask = ~torch.isnan(target)
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device, requires_grad=True)
    pred_valid = pred[valid_mask]
    target_valid = target[valid_mask]
    return nn.functional.mse_loss(pred_valid, target_valid)


# ==================== 评估函数 ====================

def evaluate_on_val_loader(model, data_loader, device):
    model.eval()
    all_true = []
    all_pred = []
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device)  # [B, H, W]，NaN 表示无效
            pred = model(x)  # [B, H, W]

            # 只取有效像素（非 NaN）
            valid = ~torch.isnan(y)
            if valid.sum() == 0:
                continue

            y_valid = y[valid].view(-1).cpu().numpy()
            p_valid = pred[valid].view(-1).cpu().numpy()
            all_true.append(y_valid)
            all_pred.append(p_valid)

    if not all_true:
        print("No valid pixels in validation set for evaluation.")
        return None

    y_concat = np.concatenate(all_true)
    p_concat = np.concatenate(all_pred)

    # 回归指标
    mse = mean_squared_error(y_concat, p_concat)
    mae = mean_absolute_error(y_concat, p_concat)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_concat, p_concat)

    print("Validation patch-level metrics (regression):")
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R²: {r2:.6f}")
    print(f"Mean True: {np.mean(y_concat):.4f}, Mean Pred: {np.mean(p_concat):.4f}")
    print(f"Std True: {np.std(y_concat):.4f}, Std Pred: {np.std(p_concat):.4f}")
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'r2': r2
    }


def predict_full_map(model, all_data_tensor, mask_tensor, target_year, window_size, patch_size=256, batch_size=1):
    """
    修复版本：处理包含NaN的patch，使用重叠滑窗和填充策略确保边缘区域也被预测
    """
    model.eval()
    H, W = mask_tensor.shape
    device = next(model.parameters()).device

    # 输出与计数 map（支持 patch 重叠平均）
    pred_sum = torch.zeros(H, W, device=device)
    pred_count = torch.zeros(H, W, device=device)

    # 计算输入数据的全局统计信息（用于填充NaN）
    valid_data = all_data_tensor[~torch.isnan(all_data_tensor)]
    fill_value = 0.0  # 默认用0填充
    if len(valid_data) > 0:
        fill_value = float(torch.nanmean(valid_data))

    with torch.no_grad():
        # 使用重叠滑窗，确保覆盖所有区域（包括边缘）
        stride = patch_size // 2  # 使用50%重叠，确保边缘也被覆盖
        for i in range(0, H, stride):
            for j in range(0, W, stride):
                i_end = min(i + patch_size, H)
                j_end = min(j + patch_size, W)
                h_cur = i_end - i
                w_cur = j_end - j

                # 如果patch完全在mask区域外，跳过（但保留边缘处理）
                patch_mask = mask_tensor[i:i_end, j:j_end]
                if (patch_mask != 1).sum() == 0:
                    continue

                # 构造输入序列 [T_w, h, w]
                x_seq_raw = all_data_tensor[target_year-window_size:target_year, i:i_end, j:j_end]  # [T_w, h, w]

                # 处理NaN值：用fill_value填充
                x_seq_filled = x_seq_raw.clone()
                nan_mask = torch.isnan(x_seq_filled)
                if nan_mask.any():
                    # 对每个时间步，尝试用最近邻有效值填充，如果没有则用全局均值
                    for t in range(x_seq_filled.shape[0]):
                        time_slice = x_seq_filled[t]
                        time_nan_mask = torch.isnan(time_slice)
                        if time_nan_mask.any():
                            # 尝试用该时间步的有效值均值填充
                            valid_values = time_slice[~time_nan_mask]
                            if len(valid_values) > 0:
                                time_fill = float(torch.mean(valid_values))
                            else:
                                time_fill = fill_value
                            x_seq_filled[t][time_nan_mask] = time_fill

                # 转换为模型输入格式 [1, T_w, 1, h, w]
                x_seq = x_seq_filled.unsqueeze(1).unsqueeze(0).to(device)  # [1, T_w, 1, h, w]

                try:
                    pred = model(x_seq)  # [1, h, w]
                    # 只对有效区域（非mask区域）累加预测值
                    valid_pred_mask = (patch_mask != 1).float().to(device)  # [h, w]
                    pred_masked = pred[0] * valid_pred_mask

                    # 对预测值做累加，用于重叠区域平均
                    pred_sum[i:i_end, j:j_end] += pred_masked
                    pred_count[i:i_end, j:j_end] += valid_pred_mask
                except Exception as e:
                    # 如果模型处理失败，跳过这个patch（但不会导致整个区域缺失，因为有重叠）
                    continue

    # 避免除以 0，只对有效区域计算平均值
    pred_count = torch.clamp(pred_count, min=1.0)
    pred_values = (pred_sum / pred_count).cpu().numpy()

    # 掩膜为 1 的地方置为 NaN
    pred_values[mask_tensor.cpu().numpy() == 1] = np.nan

    # 对于仍然为0且不在mask区域的像素（可能是没有被任何patch覆盖的边缘区域），
    # 尝试用最近邻的有效预测值填充
    valid_mask = (mask_tensor.cpu().numpy() != 1)
    pred_count_np = pred_count.cpu().numpy()
    missing_mask = valid_mask & (pred_count_np == 0)

    if missing_mask.any():
        # 使用scipy.ndimage的最近邻填充（向量化，更高效）
        try:
            from scipy.ndimage import distance_transform_edt
            pred_values_filled = pred_values.copy()

            # 找到所有有效预测值的位置
            valid_pred_mask = valid_mask & ~np.isnan(pred_values)
            if valid_pred_mask.any():
                # 使用距离变换找到最近的已知值（向量化操作）
                distances, indices = distance_transform_edt(~valid_pred_mask, return_indices=True)

                # 向量化填充：直接使用indices获取最近有效值
                # indices是一个2D数组，每个元素是最近有效值的坐标
                if len(indices) == 2:  # 2D图像
                    nearest_i, nearest_j = indices[0][missing_mask], indices[1][missing_mask]
                    # 确保索引在有效范围内
                    nearest_i = np.clip(nearest_i, 0, H-1)
                    nearest_j = np.clip(nearest_j, 0, W-1)
                    # 获取最近有效值
                    nearest_values = pred_values[nearest_i, nearest_j]
                    # 如果最近值也是NaN，使用全局均值
                    mean_value = np.nanmean(pred_values[valid_pred_mask])
                    nearest_values[np.isnan(nearest_values)] = mean_value
                    pred_values_filled[missing_mask] = nearest_values

                pred_values = pred_values_filled
        except (ImportError, Exception) as e:
            # 如果scipy不可用或出错，使用简单的均值填充
            valid_pred_mask = valid_mask & ~np.isnan(pred_values)
            if valid_pred_mask.any():
                mean_value = np.nanmean(pred_values[valid_pred_mask])
                pred_values[missing_mask] = mean_value

    return pred_values


# ==================== 主函数 ====================

def main():
    parser = argparse.ArgumentParser(description='Train ConvLSTM-UNet for spatiotemporal prediction')
    parser.add_argument('--map_type', type=str, required=True,
                        choices=['EVI', 'FVC', 'NDVI', 'RVI', 'RSEI', 'SAVI'],
                        help='Map type to train (EVI, FVC, NDVI, RVI, RSEI, SAVI)')
    parser.add_argument('--window_size', type=int, default=8,
                        help='Number of years to use as input (default: 8)')
    parser.add_argument('--patch_size', type=int, default=64,
                        help='Spatial patch size for training (default: 64)')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size (default: 128)')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--base_lr', type=float, default=1e-3,
                        help='Base learning rate (default: 1e-3)')
    parser.add_argument('--num_workers', type=int, default=2,
                        help='Number of data loading workers (default: 2)')
    parser.add_argument('--early_stop_patience', type=int, default=10,
                        help='Early stopping patience (default: 10)')
    parser.add_argument('--train_samples', type=int, default=20000,
                        help='Number of training samples (default: 20000)')
    parser.add_argument('--val_samples', type=int, default=4000,
                        help='Number of validation samples (default: 4000)')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Base data directory (default: data)')
    
    args = parser.parse_args()
    
    # 设置路径
    base_dir = Path(args.data_dir)
    npz_dir = base_dir / "interim" / "npz"
    mask_dir = base_dir / "interim"
    models_dir = Path("models")
    outputs_dir = Path("outputs") / "plots" / "ConvLSTMUNet" / args.map_type
    
    # 创建输出目录
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # ==================== 读取数据 ====================
    print("=" * 60)
    print("Loading data...")
    print("=" * 60)
    
    # 从 data/interim/npz 读取 npz 文件
    npz_path = npz_dir / f"{args.map_type}缓冲区.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    
    print(f'Loading data from: {npz_path}')
    npz_data = np.load(npz_path)
    all_data = npz_data['data']  # 形状: [T, H, W]
    
    # 加载掩码（从 pkl 文件）
    buffer_mask_path = mask_dir / "buffer_mask.pkl"
    mine_mask_path = mask_dir / "mine_mask.pkl"
    
    if not buffer_mask_path.exists():
        raise FileNotFoundError(f"Buffer mask file not found: {buffer_mask_path}")
    if not mine_mask_path.exists():
        raise FileNotFoundError(f"Mine mask file not found: {mine_mask_path}")
    
    with open(buffer_mask_path, 'rb') as f:
        buffer_mask_data = pickle.load(f)
        buffer_mask = buffer_mask_data['mask']  # uint8 格式，255 表示掩码区域
        buffer_mask = (buffer_mask == 255).astype(np.uint8)  # 转换为 0/1
    
    with open(mine_mask_path, 'rb') as f:
        mine_mask_data = pickle.load(f)
        mine_mask = mine_mask_data['mask']  # uint8 格式，255 表示掩码区域
        mine_mask = (mine_mask == 255).astype(np.uint8)  # 转换为 0/1
    
    print('Data shape:', all_data.shape)  # (T, H, W)
    print('Buffer Mask shape:', buffer_mask.shape)
    print('Buffer Unique mask values:', np.unique(buffer_mask))
    print('Mine Mask shape:', mine_mask.shape)
    print('Mine Unique mask values:', np.unique(mine_mask))
    print('Data value range:', [np.nanmin(all_data), np.nanmax(all_data)])
    
    # 基本信息（连续值预测，不需要类别数）
    T, H, W = all_data.shape
    
    # 转为 torch tensor
    all_data_tensor = torch.from_numpy(all_data).float()  # [T, H, W]
    buffer_mask_tensor = torch.from_numpy(buffer_mask).long()
    mine_mask_tensor = torch.from_numpy(mine_mask).long()
    
    print("Loading data done.")
    
    # ==================== 计算数据统计信息 ====================
    print("\n" + "=" * 60)
    print("Computing data statistics...")
    print("=" * 60)
    
    valid_mask_full = (buffer_mask != 1)
    valid_data = all_data[:, valid_mask_full]
    
    if valid_data.size > 0:
        data_mean = np.nanmean(valid_data)
        data_std = np.nanstd(valid_data)
        data_min = np.nanmin(valid_data)
        data_max = np.nanmax(valid_data)
        print(f"Data statistics (valid pixels only):")
        print(f"  Mean: {data_mean:.4f}")
        print(f"  Std: {data_std:.4f}")
        print(f"  Min: {data_min:.4f}")
        print(f"  Max: {data_max:.4f}")
    else:
        data_mean = 0.0
        data_std = 1.0
        data_min = 0.0
        data_max = 1.0
        print("Warning: No valid data found for statistics.")
    
    # ==================== 构建数据集 ====================
    print("\n" + "=" * 60)
    print("Building datasets...")
    print("=" * 60)
    
    train_dataset = SpatioTemporalPatchDataset(
        all_data_tensor, buffer_mask_tensor,
        window_size=args.window_size,
        patch_size=args.patch_size,
        num_samples=args.train_samples,
        seed=0
    )
    val_dataset = SpatioTemporalPatchDataset(
        all_data_tensor, buffer_mask_tensor,
        window_size=args.window_size,
        patch_size=args.patch_size,
        num_samples=args.val_samples,
        seed=1
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Valid years: {train_dataset.valid_years}")
    
    # ==================== 创建模型 ====================
    print("\n" + "=" * 60)
    print("Creating model...")
    print("=" * 60)
    
    model = ConvLSTMUNet().to(device)
    print(f"{model.__class__.__name__} created on {device}")
    
    # ==================== 训练 ====================
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    criterion = masked_mse_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.base_lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
    scaler = GradScaler('cuda') if device.type == 'cuda' else None
    
    best_val_loss = float('inf')
    best_epoch = -1
    patience_counter = 0
    
    train_losses_history = []
    val_losses_history = []
    
    for epoch in range(1, args.num_epochs + 1):
        model.train()
        train_loss = 0.0
        train_pixels = 0
        
        for x, y in tqdm(train_loader, desc=f"Train {epoch}/{args.num_epochs}", leave=False):
            x = x.to(device)              # [B, T, 1, P, P]
            y = y.to(device)              # [B, P, P]
            
            optimizer.zero_grad()
            
            if scaler is not None:
                with autocast('cuda'):
                    pred = model(x)  # [B, H, W]
                    loss = criterion(pred, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(x)  # [B, H, W]
                loss = criterion(pred, y)
                loss.backward()
                optimizer.step()
            
            # 只统计有效像素
            valid_pixels = (~torch.isnan(y)).sum().item()
            train_loss += loss.item() * valid_pixels
            train_pixels += valid_pixels
        
        train_loss /= max(train_pixels, 1)
        train_losses_history.append(train_loss)
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_pixels = 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Val {epoch}/{args.num_epochs}", leave=False):
                x = x.to(device)
                y = y.to(device)
                if scaler is not None:
                    with autocast('cuda'):
                        pred = model(x)  # [B, H, W]
                        loss = criterion(pred, y)
                else:
                    pred = model(x)  # [B, H, W]
                    loss = criterion(pred, y)
                # 只统计有效像素
                valid_pixels = (~torch.isnan(y)).sum().item()
                val_loss += loss.item() * valid_pixels
                val_pixels += valid_pixels
        
        val_loss /= max(val_pixels, 1)
        val_losses_history.append(val_loss)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch}/{args.num_epochs} - train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, lr: {scheduler.get_last_lr()}")
        
        # 早停与最佳模型保存
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            best_path = models_dir / f"ConvLSTMUNet_{args.map_type}_best.pth"
            torch.save(model.state_dict(), best_path)
            print(f"New best val_loss {best_val_loss:.4f} at epoch {epoch}, saved: {best_path}")
        else:
            patience_counter += 1
            print(f"patience {patience_counter} / {args.early_stop_patience}")
            if patience_counter >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch}, best epoch {best_epoch} with val_loss {best_val_loss:.4f}")
                break
    
    # ==================== 保存 loss 曲线 ====================
    print("\n" + "=" * 60)
    print("Saving loss curve...")
    print("=" * 60)
    
    # 保存loss数据
    np.savez(
        outputs_dir / f'{args.map_type}_loss_history.npz',
        train_losses=np.array(train_losses_history),
        val_losses=np.array(val_losses_history)
    )
    
    # 绘制loss曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses_history) + 1), train_losses_history, label='Train Loss')
    plt.plot(range(1, len(val_losses_history) + 1), val_losses_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss for {args.map_type}')
    plt.legend()
    plt.grid(True)
    plt.savefig(outputs_dir / f'{args.map_type}_loss_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss curve saved to: {outputs_dir / f'{args.map_type}_loss_curve.png'}")
    
    # ==================== 加载最佳模型并评估 ====================
    print("\n" + "=" * 60)
    print("Loading best model and evaluating...")
    print("=" * 60)
    
    best_path = models_dir / f"ConvLSTMUNet_{args.map_type}_best.pth"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device))
        print(f"Loaded weights from: {best_path}")
    
    # 验证集评估
    print("\nValidation set evaluation:")
    evaluate_on_val_loader(model, val_loader, device)
    
    # ==================== 整幅图预测 ====================
    print("\n" + "=" * 60)
    print("Predicting full map...")
    print("=" * 60)
    
    TARGET_YEAR_INDEX = T - 1  # 默认预测最后一年
    assert TARGET_YEAR_INDEX >= args.window_size, "TARGET_YEAR_INDEX 必须 >= WINDOW_SIZE"
    
    print(f"Predicting year index: {TARGET_YEAR_INDEX}")
    pred_map = predict_full_map(
        model, all_data_tensor, buffer_mask_tensor,
        TARGET_YEAR_INDEX, args.window_size,
        patch_size=args.patch_size, batch_size=1
    )
    print(f"Prediction map shape: {pred_map.shape}")
    
    # 选择颜色映射
    if args.map_type in ['FVC', 'EVI', 'NDVI', 'SAVI', 'RVI']:
        cmap = plt.get_cmap('RdYlGn')
    else:  # RSEI
        cmap = plt.get_cmap('viridis')
    
    # 将掩码区域显示为黑色且不进入图例
    masked_pred = np.where(buffer_mask_tensor.cpu().numpy() == 1, np.nan, pred_map)
    cmap_display = cmap.with_extremes(bad='black')  # NaN 部分渲染为黑色
    
    # 计算颜色范围（基于有效数据）
    valid_pred = pred_map[~np.isnan(pred_map)]
    vmin = np.nanmin(valid_pred) if valid_pred.size > 0 else 0
    vmax = np.nanmax(valid_pred) if valid_pred.size > 0 else 1
    
    # 绘图
    plt.figure(figsize=(10, 8))
    img = plt.imshow(masked_pred, cmap=cmap_display, vmin=vmin, vmax=vmax)
    plt.colorbar(img, label=f'{args.map_type} Value')
    plt.title(f"Predicted {args.map_type} (Continuous)", fontsize=20)
    plt.axis('off')
    
    # 保存
    plt.savefig(
        outputs_dir / f'{args.map_type}_pred_year{TARGET_YEAR_INDEX}.png',
        dpi=300, bbox_inches='tight'
    )
    plt.close()
    print(f"Prediction map saved to: {outputs_dir / f'{args.map_type}_pred_year{TARGET_YEAR_INDEX}.png'}")
    print(f"Predicted value range: [{vmin:.4f}, {vmax:.4f}]")
    
    # ==================== 评估性能指标（基于整幅预测图） ====================
    print("\n" + "=" * 60)
    print("Evaluating full map performance...")
    print("=" * 60)
    
    # 取目标年份的真实值
    true_map = all_data[TARGET_YEAR_INDEX]  # [H, W]
    
    # 有效像素：掩膜非 0，且非 NaN
    valid_mask = (buffer_mask != 1) & (~np.isnan(true_map)) & (~np.isnan(pred_map))
    
    true_flat = true_map[valid_mask].ravel()
    pred_flat = pred_map[valid_mask].ravel()
    
    print(f"Valid pixels for evaluation: {true_flat.shape[0]}")
    
    # 回归指标
    mse = mean_squared_error(true_flat, pred_flat)
    mae = mean_absolute_error(true_flat, pred_flat)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_flat, pred_flat)
    
    print(f"MSE: {mse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R²: {r2:.6f}")
    print(f"Mean True: {np.mean(true_flat):.4f}, Mean Pred: {np.mean(pred_flat):.4f}")
    print(f"Std True: {np.std(true_flat):.4f}, Std Pred: {np.std(pred_flat):.4f}")
    
    # 绘制散点图
    # 获取mine_mask在有效像素中的位置
    mine_mask_flat = mine_mask[valid_mask].ravel()
    mine_indices = mine_mask_flat == 0
    other_indices = mine_mask_flat == 1
    
    # 下采样索引（每隔1000个取一个）
    sample_indices = np.arange(len(true_flat))[::1000]
    mine_sample = mine_indices[sample_indices]
    other_sample = other_indices[sample_indices]
    
    # 计算线性回归拟合线
    lr = LinearRegression()
    lr.fit(true_flat.reshape(-1, 1), pred_flat)
    slope = lr.coef_[0]
    intercept = lr.intercept_
    
    # 计算拟合线的预测值
    x_line = np.linspace(true_flat.min(), true_flat.max(), 100)
    y_line = slope * x_line + intercept
    
    # 计算置信区间（95%置信区间）
    residuals = pred_flat - (slope * true_flat + intercept)
    n = len(true_flat)
    dof = n - 2  # 自由度
    mse_residual = np.sum(residuals**2) / dof  # 均方误差
    std_residual = np.sqrt(mse_residual)
    
    # 计算每个x值对应的置信区间
    t_critical = stats.t.ppf(0.975, dof)  # 95%置信区间，t分布
    x_mean = np.mean(true_flat)
    sxx = np.sum((true_flat - x_mean)**2)
    
    # 对x_line中的每个点计算预测值的标准误差
    se_pred = std_residual * np.sqrt(1 + 1/n + (x_line - x_mean)**2 / sxx)
    ci_upper = y_line + t_critical * se_pred
    ci_lower = y_line - t_critical * se_pred
    
    # 绘制散点图（真实值 vs 预测值）
    plt.figure(figsize=(10, 10))
    
    # 绘制置信区间（半透明红色带状区域）
    plt.fill_between(x_line, ci_lower, ci_upper, alpha=0.1, color='red', label='95% Confidence Interval')
    
    # 绘制散点：mine_mask区域内的点显示为红色，其他显示为蓝色
    if mine_sample.sum() > 0:
        plt.scatter(true_flat[sample_indices][mine_sample], pred_flat[sample_indices][mine_sample],
                    alpha=0.1, s=10, c='red', label='Mine Region')
    if other_sample.sum() > 0:
        plt.scatter(true_flat[sample_indices][other_sample], pred_flat[sample_indices][other_sample],
                    alpha=0.1, s=10, c='blue', label='Buffer Region')
    
    # 绘制理想预测线（y=x，虚线）
    plt.plot([true_flat.min(), true_flat.max()], [true_flat.min(), true_flat.max()],
             'k--', lw=2.5, label='Perfect Prediction (y=x)', zorder=5)
    
    # 绘制拟合回归线（实线）
    plt.plot(x_line, y_line, 'b-', lw=2, label=f'Regression Line (y={slope:.3f}x+{intercept:.3f})', zorder=4)
    
    plt.xlabel('True Value', fontsize=23)
    plt.ylabel('Predicted Value', fontsize=23)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(f'True vs Predicted ({args.map_type})', fontsize=25)
    plt.legend(fontsize=20, loc='best')
    plt.axis('equal')  # 保持坐标轴比例一致
    
    plt.savefig(
        outputs_dir / f'{args.map_type}_scatter_year{TARGET_YEAR_INDEX}.png',
        dpi=300, bbox_inches='tight'
    )
    plt.close()
    print(f"Scatter plot saved to: {outputs_dir / f'{args.map_type}_scatter_year{TARGET_YEAR_INDEX}.png'}")
    
    # 打印回归线信息
    print(f"\nRegression Line: y = {slope:.6f}x + {intercept:.6f}")
    print(f"R² Score: {r2:.6f}")
    
    print("\n" + "=" * 60)
    print("Training and evaluation completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
