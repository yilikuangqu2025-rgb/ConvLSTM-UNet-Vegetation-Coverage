"""
处理 raw_data 子目录下的所有 TIF 文件
将每个子目录的时间维数据组合成三维数组，并保存为 npz 文件
"""
import re
from pathlib import Path
import numpy as np
import pickle
from read_tif import read_singleband_tif, handle_nan_values
from tqdm import tqdm


def extract_year_from_filename(filename):
    """从文件名中提取年份"""
    # 匹配文件名中的4位数字年份（2000-2099）
    match = re.search(r'(\d{4})', filename)
    if match:
        return int(match.group(1))
    return None


def process_subdirectory(subdir_path, output_dir, extra_nodata=None):
    """
    处理单个子目录下的所有 TIF 文件
    
    参数:
        subdir_path: 子目录路径
        output_dir: 输出目录
        extra_nodata: 额外的 nodata 值
    """
    subdir_path = Path(subdir_path)
    if not subdir_path.exists() or not subdir_path.is_dir():
        print(f"警告: 目录不存在或不是目录: {subdir_path}")
        return
    
    # 查找所有 TIF 文件
    tif_files = list(subdir_path.glob("*.tif"))
    if not tif_files:
        print(f"警告: 在 {subdir_path} 中未找到 TIF 文件")
        return
    
    # 提取年份并排序
    file_year_pairs = []
    for tif_file in tif_files:
        year = extract_year_from_filename(tif_file.name)
        if year is not None:
            file_year_pairs.append((year, tif_file))
        else:
            print(f"警告: 无法从文件名提取年份: {tif_file.name}")
    
    if not file_year_pairs:
        print(f"警告: 在 {subdir_path} 中未找到有效的年份信息")
        return
    
    # 按年份排序
    file_year_pairs.sort(key=lambda x: x[0])
    
    print(f"\n处理目录: {subdir_path.name}")
    print(f"找到 {len(file_year_pairs)} 个 TIF 文件")
    print(f"年份范围: {file_year_pairs[0][0]} - {file_year_pairs[-1][0]}")
    
    # 读取第一个文件以确定尺寸
    first_year, first_file = file_year_pairs[0]
    first_data, first_valid_mask = read_singleband_tif(first_file, extra_nodata=extra_nodata, show_info=False)
    height, width = first_data.shape
    
    print(f"栅格尺寸: {height} x {width}")
    
    # 根据子目录名称加载相应的掩码
    mask = None
    subdir_name = subdir_path.name
    interim_dir = Path("data") / "interim"
    
    if "缓冲区" in subdir_name:
        mask_path = interim_dir / "buffer_mask.pkl"
        if mask_path.exists():
            print(f"加载缓冲区掩码: {mask_path}")
            with open(mask_path, "rb") as f:
                mask_data = pickle.load(f)
                mask = mask_data['mask']  # uint8 格式，255 表示掩码区域
                # 转换为布尔掩码（255 -> True, 0 -> False）
                mask = (mask == 255)
                print(f"  掩码形状: {mask.shape}")
                # 检查掩码尺寸是否匹配
                if mask.shape != (height, width):
                    print(f"  警告: 掩码尺寸 {mask.shape} 与数据尺寸 {(height, width)} 不匹配，跳过掩码应用")
                    mask = None
                else:
                    mask_pixels = mask.sum()
                    print(f"  掩码区域像元数: {mask_pixels:,}")
        else:
            print(f"  警告: 未找到掩码文件 {mask_path}")
    
    elif "矿" in subdir_name:
        mask_path = interim_dir / "mine_mask.pkl"
        if mask_path.exists():
            print(f"加载矿掩码: {mask_path}")
            with open(mask_path, "rb") as f:
                mask_data = pickle.load(f)
                mask = mask_data['mask']  # uint8 格式，255 表示掩码区域
                # 转换为布尔掩码（255 -> True, 0 -> False）
                mask = (mask == 255)
                print(f"  掩码形状: {mask.shape}")
                # 检查掩码尺寸是否匹配
                if mask.shape != (height, width):
                    print(f"  警告: 掩码尺寸 {mask.shape} 与数据尺寸 {(height, width)} 不匹配，跳过掩码应用")
                    mask = None
                else:
                    mask_pixels = mask.sum()
                    print(f"  掩码区域像元数: {mask_pixels:,}")
        else:
            print(f"  警告: 未找到掩码文件 {mask_path}")
    
    # 初始化三维数组 (时间, 高度, 宽度)
    num_times = len(file_year_pairs)
    data_array = np.zeros((num_times, height, width), dtype=np.float32)
    valid_mask_array = np.zeros((num_times, height, width), dtype=bool)
    years = []
    
    # 读取所有文件
    for idx, (year, tif_file) in tqdm(list(enumerate(file_year_pairs)), total=len(file_year_pairs), desc="读取tif文件", leave=False):
        # print(f"\r  读取 [{idx+1}/{num_times}]: {tif_file.name} (年份: {year})", end="")
        try:
            data, valid_mask = read_singleband_tif(tif_file, extra_nodata=extra_nodata, show_info=False)
            
            # 处理空值（NaN）
            num_nan_before = np.isnan(data).sum()
            if num_nan_before > 0:
                # 插值处理空值
                # print(f"\n    检测到空值 (NaN): {num_nan_before:,} 个")
                data, nan_mask, nan_count = handle_nan_values(data, method='interpolate')
                # print(f"    已处理空值: {nan_count:,} 个")
            
            # 检查尺寸是否一致
            if data.shape != (height, width):
                print(f"  警告: 文件 {tif_file.name} 的尺寸 {data.shape} 与第一个文件不一致，跳过")
                continue
            
            # 应用掩码：将掩码区域的值设置为 None
            if mask is not None:
                data[mask] = None
            
            data_array[idx] = data
            valid_mask_array[idx] = valid_mask
            years.append(year)
        except Exception as e:
            print(f"  错误: 读取文件 {tif_file.name} 时出错: {e}")
            continue
    
    if len(years) == 0:
        print(f"错误: 未能成功读取任何文件")
        return
    
    # 如果有些文件读取失败，调整数组大小
    if len(years) < num_times:
        data_array = data_array[:len(years)]
        valid_mask_array = valid_mask_array[:len(years)]
    
    # 保存为 npz 文件
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 使用子目录名称作为输出文件名
    output_filename = output_dir / f"{subdir_path.name}.npz"
    
    np.savez_compressed(
        output_filename,
        data=data_array,
        valid_mask=valid_mask_array,
        years=np.array(years),
        shape=data_array.shape,
        height=height,
        width=width,
        num_times=len(years)
    )
    
    print(f"\r  已保存: {output_filename}")
    print(f"  数组形状: {data_array.shape}")
    print(f"  年份: {years[0]} - {years[-1]}")
    print(f"  有效像元统计: {valid_mask_array.sum()} / {valid_mask_array.size}")


def main():
    """主函数：处理 raw_data 下的所有子目录"""
    base_dir = Path("data") / "raw_data"
    output_dir = Path("data") / "interim" / "npz"
    
    if not base_dir.exists():
        raise FileNotFoundError(f"未找到数据目录: {base_dir}")
    
    # 获取所有子目录
    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
    
    if not subdirs:
        print(f"警告: 在 {base_dir} 中未找到子目录")
        return
    
    print(f"找到 {len(subdirs)} 个子目录")
    print("=" * 60)
    
    # 处理每个子目录
    for subdir in subdirs:
        try:
            process_subdirectory(subdir, output_dir, extra_nodata=0)
        except Exception as e:
            print(f"错误: 处理目录 {subdir.name} 时出错: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("处理完成！")


if __name__ == "__main__":
    main()

