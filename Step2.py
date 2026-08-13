import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
from scipy.ndimage import zoom


def save_to_npy(data, file_path):
    """保存为NPY格式"""
    np.save(file_path, data)
    print(f"  ✓ NPY: {os.path.basename(file_path)}")


def save_to_bin(data, file_path):
    """保存为二进制格式"""
    data.astype(np.float32).tofile(file_path)
    print(f"  ✓ BIN: {os.path.basename(file_path)}")


def save_to_mat(data, file_path):
    """保存为MATLAB格式"""
    from scipy.io import savemat
    savemat(file_path, {'data': data})
    print(f"  ✓ MAT: {os.path.basename(file_path)}")


def save_to_dat(data, file_path):
    """保存为DAT文本格式"""
    np.savetxt(file_path, data, fmt='%.6f', delimiter='\t')
    print(f"  ✓ DAT: {os.path.basename(file_path)}")


def save_to_txt(data, file_path):
    """保存为TXT格式"""
    np.savetxt(file_path, data, fmt='%.6f', delimiter='\t')
    print(f"  ✓ TXT: {os.path.basename(file_path)}")


def save_to_csv(data, file_path):
    """保存为CSV格式"""
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False, header=False)
    print(f"  ✓ CSV: {os.path.basename(file_path)}")


def load_data_memory_efficient(data_dir, base_name, data_shape):
    """
    内存高效地加载所有格式的数据
    """
    data_dict = {}

    # 加载NPY（使用内存映射）
    npy_path = os.path.join(data_dir, f'{base_name}.npy')
    if os.path.exists(npy_path):
        try:
            data_dict['npy'] = np.load(npy_path, mmap_mode='r')
            print(f"    NPY (mmap): {data_dict['npy'].shape}")
        except Exception as e:
            print(f"    加载NPY失败: {e}")
            data_dict['npy'] = None
    else:
        data_dict['npy'] = None

    # 加载BIN（使用内存映射）
    bin_path = os.path.join(data_dir, f'{base_name}.bin')
    if os.path.exists(bin_path):
        try:
            if data_shape is not None:
                data_dict['bin'] = np.memmap(bin_path, dtype=np.float32, mode='r', shape=data_shape)
                print(f"    BIN (mmap): {data_dict['bin'].shape}")
            else:
                data_dict['bin'] = None
        except Exception as e:
            print(f"    加载BIN失败: {e}")
            data_dict['bin'] = None
    else:
        data_dict['bin'] = None

    # 加载MAT
    mat_path = os.path.join(data_dir, f'{base_name}.mat')
    if os.path.exists(mat_path):
        try:
            mat_data = loadmat(mat_path)
            data_dict['mat'] = mat_data['data']
            print(f"    MAT: {data_dict['mat'].shape}")
        except Exception as e:
            print(f"    加载MAT失败: {e}")
            data_dict['mat'] = None
    else:
        data_dict['mat'] = None

    # 加载DAT
    dat_path = os.path.join(data_dir, f'{base_name}.dat')
    if os.path.exists(dat_path):
        try:
            data_dict['dat'] = np.loadtxt(dat_path)
            print(f"    DAT: {data_dict['dat'].shape}")
        except Exception as e:
            print(f"    加载DAT失败: {e}")
            data_dict['dat'] = None
    else:
        data_dict['dat'] = None

    # 加载TXT
    txt_path = os.path.join(data_dir, f'{base_name}.txt')
    if os.path.exists(txt_path):
        try:
            data_dict['txt'] = np.loadtxt(txt_path)
            print(f"    TXT: {data_dict['txt'].shape}")
        except Exception as e:
            print(f"    加载TXT失败: {e}")
            data_dict['txt'] = None
    else:
        data_dict['txt'] = None

    # 加载CSV
    csv_path = os.path.join(data_dir, f'{base_name}.csv')
    if os.path.exists(csv_path):
        try:
            csv_data = pd.read_csv(csv_path, header=None)
            data_dict['csv'] = csv_data.values
            print(f"    CSV: {data_dict['csv'].shape}")
        except Exception as e:
            print(f"    加载CSV失败: {e}")
            data_dict['csv'] = None
    else:
        data_dict['csv'] = None

    return data_dict


def main():
    # 配置
    data_dir = 'data'
    output_dir = 'output_data'
    fig_dir = os.path.join(output_dir, 'figs')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    # 文件路径
    mat_file = os.path.join(data_dir, 'MODEL_DENSITY_1.25m.mat')
    base_name = 'MODEL_DENSITY_1.25m_resampled_10x'

    print("=" * 60)
    print("开始处理 MODEL_DENSITY_1.25m.mat")
    print("=" * 60)

    # 1. 加载原始数据
    print("\n[1] 加载原始数据...")
    start_time = time.time()

    try:
        mat_data = loadmat(mat_file)
        # 查找数据变量（通常是'data'或第一个非元数据变量）
        data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
        if data_keys:
            data_key = data_keys[0]
            original_data = mat_data[data_key]
            print(f"  从MAT文件加载: 变量名 '{data_key}', 形状 {original_data.shape}")
        else:
            raise ValueError("未找到数据变量")

        # 确保数据是2D的
        if original_data.ndim > 2:
            original_data = original_data.squeeze()

        print(f"  原始数据形状: {original_data.shape}")
        print(f"  原始数据范围: [{original_data.min():.2f}, {original_data.max():.2f}]")

    except Exception as e:
        print(f"  加载失败: {e}")
        return

    # 2. 10倍重采样
    print("\n[2] 10倍重采样...")

    # 使用scipy的zoom函数进行重采样（比切片更平滑）
    # 目标形状: 2801/10 ≈ 281, 13601/10 ≈ 1361
    original_shape = original_data.shape
    # 计算目标形状（四舍五入）
    target_shape = (int(round(original_shape[0] / 10)), int(round(original_shape[1] / 10)))
    print(f"  原始形状: {original_shape}")
    print(f"  目标形状: {target_shape}")
    print(f"  实际缩放因子: {target_shape[0] / original_shape[0]:.4f}, {target_shape[1] / original_shape[1]:.4f}")

    # 使用zoom进行重采样（更高质量）
    zoom_factors = (target_shape[0] / original_shape[0], target_shape[1] / original_shape[1])
    resampled_data = zoom(original_data, zoom_factors, order=1)  # order=1为双线性插值

    # 或者使用简单切片（更快但可能产生混叠）
    # resampled_data = original_data[::10, ::10]

    print(f"  重采样后形状: {resampled_data.shape}")
    print(f"  重采样后范围: [{resampled_data.min():.2f}, {resampled_data.max():.2f}]")
    print(f"  重采样耗时: {time.time() - start_time:.2f}秒")

    # 3. 保存为多种格式
    print("\n[3] 保存为多种格式...")
    save_start = time.time()

    # NPY
    save_to_npy(resampled_data, os.path.join(output_dir, f'{base_name}.npy'))

    # BIN
    save_to_bin(resampled_data, os.path.join(output_dir, f'{base_name}.bin'))

    # MAT
    save_to_mat(resampled_data, os.path.join(output_dir, f'{base_name}.mat'))

    # DAT
    save_to_dat(resampled_data, os.path.join(output_dir, f'{base_name}.dat'))

    # TXT
    save_to_txt(resampled_data, os.path.join(output_dir, f'{base_name}.txt'))

    # CSV
    save_to_csv(resampled_data, os.path.join(output_dir, f'{base_name}.csv'))

    print(f"  保存耗时: {time.time() - save_start:.2f}秒")

    # 4. 加载所有格式并绘图
    print("\n[4] 加载所有格式并绘制对比图...")

    # 使用内存映射加载
    data_dict = load_data_memory_efficient(output_dir, base_name, resampled_data.shape)

    # 检查哪些格式可用
    available_formats = [fmt for fmt, data in data_dict.items() if data is not None]
    print(f"  可用格式: {', '.join([f.upper() for f in available_formats])}")

    if not available_formats:
        print("  错误: 没有可用数据")
        return

    # 绘图
    print("\n[5] 生成对比图...")

    format_labels = {
        'npy': 'NPY (mmap)',
        'bin': 'BIN (mmap)',
        'mat': 'MAT',
        'dat': 'DAT',
        'txt': 'TXT',
        'csv': 'CSV'
    }

    n_formats = len(available_formats)
    n_cols = min(3, n_formats)
    n_rows = (n_formats + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

    # 确保axes是1D数组
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    plot_idx = 0
    for fmt in available_formats:
        data = data_dict[fmt]

        # 显示2D数据
        im = axes[plot_idx].imshow(data, aspect='auto', cmap='jet', interpolation='bilinear')
        axes[plot_idx].set_title(f'{format_labels.get(fmt, fmt.upper())}\n{data.shape}',
                                 fontsize=13, fontweight='bold')
        axes[plot_idx].set_xlabel('X (Trace)', fontsize=10)
        axes[plot_idx].set_ylabel('Y (Sample)', fontsize=10)
        axes[plot_idx].tick_params(labelsize=9)

        cbar = plt.colorbar(im, ax=axes[plot_idx])
        cbar.ax.tick_params(labelsize=9)
        cbar.set_label('Density', fontsize=9)

        # 显示数据统计信息
        info_text = f'Min: {data.min():.2f}\nMax: {data.max():.2f}\nMean: {data.mean():.2f}'
        axes[plot_idx].text(0.02, 0.98, info_text,
                            transform=axes[plot_idx].transAxes,
                            fontsize=8,
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # 标记内存映射格式
        if fmt in ['npy', 'bin']:
            axes[plot_idx].text(0.98, 0.02, '🔹 mmap',
                                transform=axes[plot_idx].transAxes,
                                fontsize=8, horizontalalignment='right',
                                verticalalignment='bottom',
                                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

        plot_idx += 1

    # 隐藏多余的子图
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'{base_name}\nMulti-Format Comparison (10x Downsampled)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    fig_path = os.path.join(fig_dir, f'{base_name}_comparison.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ 对比图已保存: {fig_path}")
    plt.close()

    # 总结
    print("\n" + "=" * 60)
    print("处理完成！")
    print("=" * 60)
    print(f"数据目录: {output_dir}")
    print(f"对比图: {fig_path}")
    print(f"重采样后形状: {resampled_data.shape}")
    print(f"总耗时: {time.time() - start_time:.2f}秒")
    print("=" * 60)


if __name__ == "__main__":
    main()


