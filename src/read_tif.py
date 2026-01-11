import rasterio
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
# import geopandas as gpd


def read_singleband_tif(tif_path, extra_nodata=None, show_info=True):
    """
    单波段 TIF 读取函数：
    把栅格自身的 nodata, 以及 extra_nodata 中指定的值视为无效值
    返回: data(二维float 数组)、valid_mask(True 表示有效像元)
    """
    tif_path = Path(tif_path)
    if not tif_path.exists():
        raise FileNotFoundError(f"未找到栅格文件: {tif_path}")

    with rasterio.open(tif_path) as src:
        if show_info:
            print("="*50)
            print(f"读取文件信息: {tif_path}")
            print(f"栅格尺寸 (宽, 高): {src.width}, {src.height}")
            print(f"波段数: {src.count}")
            print(f"坐标系 (CRS): {src.crs}")
            print(f"像元大小: {src.res}")
            print(f"有效像元数量: {src.width * src.height}")
            print("="*50)
        data = src.read(1).astype(float)
        raw_nodata = src.nodata

    # 处理nodata值
    nodata_vals = []
    if raw_nodata is not None:
        nodata_vals.append(raw_nodata)
    if extra_nodata is not None:
        if isinstance(extra_nodata, (list, tuple, np.ndarray)):
            nodata_vals.extend(list(extra_nodata))
        else:
            nodata_vals.append(extra_nodata)

    if nodata_vals:
        invalid_mask = np.isin(data, nodata_vals)
    else:
        invalid_mask = np.zeros_like(data, dtype=bool)

    valid_mask = ~invalid_mask
    return data, valid_mask


def handle_nan_values(data, method='fill', fill_value=-9999, interpolation_method='nearest'):
    """
    处理二维数组中的空值（NaN）
    
    参数:
        data: 二维 numpy 数组，可能包含 NaN 值
        method: 处理方法，可选值：
            - 'fill': 用指定值填充（默认 -9999）
            - 'interpolate': 使用插值填充（需要 scipy）
            - 'forward_fill': 前向填充（用前一个有效值填充）
            - 'backward_fill': 后向填充（用后一个有效值填充）
            - 'mean': 用有效值的均值填充
            - 'median': 用有效值的中位数填充
            - 'zero': 用 0 填充
            - 'remove': 返回时保持 NaN（不处理）
        fill_value: 当 method='fill' 时使用的填充值（默认 -9999）
        interpolation_method: 当 method='interpolate' 时使用的插值方法（默认 'nearest'）
    
    返回:
        processed_data: 处理后的二维数组
        nan_mask: 布尔数组，True 表示原始 NaN 位置
        nan_count: NaN 值的数量
    """
    data = np.asarray(data, dtype=np.float32)
    nan_mask = np.isnan(data)
    nan_count = nan_mask.sum()
    
    if nan_count == 0:
        return data, nan_mask, nan_count
    
    processed_data = data.copy()
    
    if method == 'fill':
        # 用指定值填充
        processed_data[nan_mask] = fill_value
        
    elif method == 'zero':
        # 用 0 填充
        processed_data[nan_mask] = 0.0
        
    elif method == 'mean':
        # 用有效值的均值填充
        valid_values = data[~nan_mask]
        if valid_values.size > 0:
            mean_value = np.mean(valid_values)
            processed_data[nan_mask] = mean_value
        else:
            print("警告: 所有值都是 NaN，无法计算均值")
            
    elif method == 'median':
        # 用有效值的中位数填充
        valid_values = data[~nan_mask]
        if valid_values.size > 0:
            median_value = np.median(valid_values)
            processed_data[nan_mask] = median_value
        else:
            print("警告: 所有值都是 NaN，无法计算中位数")
            
    elif method == 'interpolate':
        # 使用插值填充（需要 scipy）
        try:
            from scipy.interpolate import griddata
            
            # 获取有效值的位置和值
            valid_mask = ~nan_mask
            rows, cols = np.where(valid_mask)
            values = data[valid_mask]
            
            # 获取 NaN 位置
            nan_rows, nan_cols = np.where(nan_mask)
            
            if len(rows) > 0 and len(nan_rows) > 0:
                # 创建坐标点
                valid_points = np.column_stack((rows, cols))
                nan_points = np.column_stack((nan_rows, nan_cols))
                
                # 插值
                interpolated_values = griddata(
                    valid_points, values, nan_points,
                    method=interpolation_method,
                    fill_value=np.nanmean(values) if values.size > 0 else 0
                )
                
                processed_data[nan_mask] = interpolated_values
            else:
                print("警告: 没有足够的有效值进行插值")
        except ImportError:
            print("警告: scipy 未安装，无法使用插值方法，改用均值填充")
            valid_values = data[~nan_mask]
            if valid_values.size > 0:
                processed_data[nan_mask] = np.mean(valid_values)
                
    elif method == 'forward_fill':
        # 前向填充（按行）
        for i in range(processed_data.shape[0]):
            row = processed_data[i, :]
            nan_indices = np.isnan(row)
            if nan_indices.any():
                # 找到第一个有效值
                valid_indices = np.where(~nan_indices)[0]
                if len(valid_indices) > 0:
                    # 用第一个有效值填充前面的 NaN
                    first_valid_idx = valid_indices[0]
                    first_valid_value = row[first_valid_idx]
                    row[nan_indices & (np.arange(len(row)) < first_valid_idx)] = first_valid_value
                    # 前向填充
                    for j in range(len(row)):
                        if np.isnan(row[j]) and j > 0:
                            row[j] = row[j-1]
                processed_data[i, :] = row
                
    elif method == 'backward_fill':
        # 后向填充（按行）
        for i in range(processed_data.shape[0]):
            row = processed_data[i, :]
            nan_indices = np.isnan(row)
            if nan_indices.any():
                # 找到最后一个有效值
                valid_indices = np.where(~nan_indices)[0]
                if len(valid_indices) > 0:
                    # 用最后一个有效值填充后面的 NaN
                    last_valid_idx = valid_indices[-1]
                    last_valid_value = row[last_valid_idx]
                    row[nan_indices & (np.arange(len(row)) > last_valid_idx)] = last_valid_value
                    # 后向填充
                    for j in range(len(row)-2, -1, -1):
                        if np.isnan(row[j]):
                            row[j] = row[j+1]
                processed_data[i, :] = row
                
    elif method == 'remove':
        # 保持 NaN，不处理
        pass
        
    else:
        raise ValueError(f"未知的处理方法: {method}")
    
    return processed_data, nan_mask, nan_count


if __name__ == "__main__":
    base_dir = Path("data") / "raw_data" / "FVC缓冲区"

    tif_path = base_dir / "Landsat_FVC_2001.tif"

    # 函数读取tif，将-9999也视为nodata
    data, valid_mask = read_singleband_tif(tif_path)
    print(data)
    
    
    num_nan_before = np.isnan(data).sum()
    if num_nan_before > 0:
        print(f"\n    检测到空值 (NaN): {num_nan_before:,} 个")
        # 使用 fill 方法，将 NaN 替换为 -9999（与掩码处理一致）
        data, nan_mask, nan_count = handle_nan_values(data, method='interpolate')
        print(f"    已处理空值: {nan_count:,} 个 NaN 值已替换为 -9999")

    valid = data[valid_mask]
    if valid.size > 0:
        print(f"最小值: {float(np.nanmin(valid))}")
        print(f"最大值: {float(np.nanmax(valid))}")
        print(f"平均值: {float(np.nanmean(valid))}")

    # 打开栅格以获取空间范围和坐标系
    with rasterio.open(tif_path) as src:
        bounds = src.bounds  # left, bottom, right, top
        raster_crs = src.crs

    # 可视化：栅格 + 矿集区边界 + 每个区域编号
    masked_data = np.ma.masked_array(data, mask=~valid_mask)

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="white")

    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(
        masked_data,
        cmap=cmap,
        extent=(bounds.left, bounds.right, bounds.bottom, bounds.top),
        origin="upper",
    )

    ax.set_title("FVC_2000 with Mining Areas")
    plt.colorbar(im, ax=ax, label="FVC")
    ax.set_axis_off()

    # 输出到文件
    out_png = base_dir / "EVI_2000.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    print(f"热力图已保存: {out_png}")
    plt.close()

