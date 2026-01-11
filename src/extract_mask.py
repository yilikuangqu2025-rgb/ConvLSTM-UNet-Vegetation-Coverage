"""
读取 TIF 文件，提取数值为 0 的区域 mask 并保存为 pkl 文件
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pickle
from read_tif import read_singleband_tif
import os

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

def save_zero_mask_to_pkl(tif_path, output_path, extra_nodata=None):
    """
    读取 TIF 文件，提取数值为 0 的区域 mask 并保存为 pkl 文件
    
    参数:
        tif_path: TIF 文件路径
        output_path: 输出 pkl 文件路径
        extra_nodata: 额外的 nodata 值
    
    返回:
        mask_uint8: uint8 格式的掩码数组
    """
    tif_path = Path(tif_path)
    output_path = Path(output_path)
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 读取 TIF 文件
    print(f"读取文件: {tif_path}")
    data, valid_mask = read_singleband_tif(tif_path, extra_nodata=extra_nodata, show_info=True)
    
    # 创建数值为 0 的掩码
    zero_mask = (data == 0)
    
    # 统计信息
    total_pixels = data.size
    zero_pixels = zero_mask.sum()
    zero_percentage = (zero_pixels / total_pixels) * 100
    
    print(f"\n统计信息:")
    print(f"  总像元数: {total_pixels:,}")
    print(f"  数值为 0 的像元数: {zero_pixels:,}")
    print(f"  占比: {zero_percentage:.2f}%")
    
    # 将 mask 转换为 uint8 格式（0 或 255），参考 extract_mask.py 的格式
    mask_uint8 = zero_mask.astype(np.uint8) * 255
    
    # 存储掩码数据，参考 extract_mask.py 的格式
    mask_data = {
        'mask': mask_uint8,
        'original_shape': data.shape,
    }
    
    # 保存到 pkl 文件
    with open(output_path, "wb") as f:
        pickle.dump(mask_data, f)
    
    print(f"掩码已保存到: {output_path}")
    print(f"  mask 形状: {mask_uint8.shape}")
    print(f"  mask 数据类型: {mask_uint8.dtype}")
    print(f"  原始数据形状: {data.shape}")
    
    return mask_uint8


if __name__ == "__main__":
    # 定义基础路径
    base_dir = Path("data")
    raw_data_dir = base_dir / "raw_data"
    interim_dir = base_dir / "interim"
    
    # 确保输出目录存在
    interim_dir.mkdir(parents=True, exist_ok=True)
    
    buffer_mask = None
    mine_mask = None
    
    # 处理 EVI缓冲区
    buffer_tif_path = raw_data_dir / "EVI缓冲区" / "Landsat_EVI_2000.tif"
    buffer_output_path = interim_dir / "buffer_mask.pkl"
    
    if buffer_tif_path.exists():
        print("=" * 60)
        print("处理 EVI缓冲区")
        print("=" * 60)
        buffer_mask = save_zero_mask_to_pkl(buffer_tif_path, buffer_output_path, extra_nodata=0)
    else:
        print(f"警告: 未找到文件 {buffer_tif_path}")
    
    # 处理 EVI矿
    mine_tif_path = raw_data_dir / "EVI矿" / "Landsat_EVI_2000.tif"
    mine_output_path = interim_dir / "mine_mask.pkl"
    
    if mine_tif_path.exists():
        print("\n" + "=" * 60)
        print("处理 EVI矿")
        print("=" * 60)
        mine_mask = save_zero_mask_to_pkl(mine_tif_path, mine_output_path, extra_nodata=0)
    else:
        print(f"警告: 未找到文件 {mine_tif_path}")
    
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)
    
    # 显示两个掩码
    if buffer_mask is not None or mine_mask is not None:
        print("\n显示掩码...")
        
        # 创建子图
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # 显示 EVI缓冲区掩码
        if buffer_mask is not None:
            ax = axes[0]
            im = ax.imshow(buffer_mask, cmap='Reds', origin='upper')
            ax.set_title("EVI缓冲区 - 数值为 0 的区域掩码", fontsize=14)
            ax.set_xlabel("列", fontsize=12)
            ax.set_ylabel("行", fontsize=12)
            plt.colorbar(im, ax=ax, label="掩码值 (255=数值为0)")
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        else:
            axes[0].axis('off')
            axes[0].text(0.5, 0.5, '未找到数据', ha='center', va='center', transform=axes[0].transAxes)
        
        # 显示 EVI矿掩码
        if mine_mask is not None:
            ax = axes[1]
            im = ax.imshow(mine_mask, cmap='Reds', origin='upper')
            ax.set_title("EVI矿 - 数值为 0 的区域掩码", fontsize=14)
            ax.set_xlabel("列", fontsize=12)
            ax.set_ylabel("行", fontsize=12)
            plt.colorbar(im, ax=ax, label="掩码值 (255=数值为0)")
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        else:
            axes[1].axis('off')
            axes[1].text(0.5, 0.5, '未找到数据', ha='center', va='center', transform=axes[1].transAxes)
        
        plt.tight_layout()
        plt.show()


