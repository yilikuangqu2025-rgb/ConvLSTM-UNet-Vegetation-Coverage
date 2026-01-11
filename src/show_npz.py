import numpy as np
import matplotlib.pyplot as plt
import os
import math

plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# ========= 参数 =========
npz_dir_path = 'data/interim/npz'          # npz 文件路径
output_dir = 'data/interim/imgs'           # 输出目录
cmap = 'viridis'                           # 可换 'gray' / 'jet' 等
dpi = 200

os.makedirs(output_dir, exist_ok=True)

for npz_file in os.listdir(npz_dir_path):
    npz_path = os.path.join(npz_dir_path, npz_file)
    if not npz_path.endswith('.npz'):
        continue
    
    # ========= 读取数据 =========
    data = np.load(npz_path)['data']   # shape: (26, H, W)
    print(npz_file, 'Data shape:', data.shape)

    # 统一的最小值和最大值（忽略 NaN）
    vmin = np.nanmin(data)
    vmax = np.nanmax(data)

    n_slices = data.shape[0]
    n_cols = 4
    n_rows = math.ceil(n_slices / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 2 * n_rows), gridspec_kw={'hspace': 0, 'wspace': 0.02})
    axes = np.array(axes).reshape(n_rows, n_cols)

    last_im = None

    # ========= 逐张可视化到同一副图 =========
    for i in range(n_slices):
        img = data[i]
        nan_pixels = np.sum(np.isnan(img))
        neg_pixels = np.sum(img == -9999)
        
        mask_count = None
        if "缓冲区" in npz_path:
            mask_count = 2403682
        elif "矿" in npz_path:
            mask_count = 4280139
        
        if nan_pixels != mask_count or neg_pixels:
            print(f'\tSlice {i}: NaN pixels = {nan_pixels}')
            print(f'\tSlice {i}: -9999 pixels = {neg_pixels}')

        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]

        last_im = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(f'Slice {i}', fontsize=10)
        # ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])
        # 显示边框
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('gray')
            spine.set_linewidth(0.5)

    # 关闭空子图（但保留边框）
    for j in range(n_slices, n_rows * n_cols):
        row = j // n_cols
        col = j % n_cols
        axes[row, col].set_xticks([])
        axes[row, col].set_yticks([])
        # 显示边框
        for spine in axes[row, col].spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('lightgray')
            spine.set_linewidth(0.5)

    # 统一颜色条 - 控制位置
    if last_im is not None:
        # 使用 fig.colorbar 的 location 参数控制位置，或使用 cax 参数精确控制
        # 在右侧添加 colorbar，占用整个图形高度的 80%
        # 放在左下角
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        # 选择左下角的子图
        ax_bottom_left = axes[-1, -2]

        # 使用axes_grid1的make_axes_locatable在左下角贴一个colorbar
        divider = make_axes_locatable(ax_bottom_left)
        cax = divider.append_axes("top", size="50%", pad=0.5)

        cbar = fig.colorbar(last_im, cax=cax, orientation='horizontal')
        cbar.ax.set_ylabel('Value', fontsize=10)

    plt.suptitle(npz_file, fontsize=14)
    # 使用 subplots_adjust 手动控制整体布局，避免 tight_layout 的兼容性 warning
    fig.subplots_adjust(
        left=0.05, right=0.98,
        bottom=0.05, top=0.90,
        hspace=0.4,  # 垂直间距
        wspace=0.0   # 水平间距
    )
    # plt.show()

    save_path = os.path.join(output_dir, f'{os.path.splitext(npz_file)[0]}.png')
    plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

print('Done! Images saved to:', output_dir)