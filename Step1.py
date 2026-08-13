import numpy as np
import obspy
from obspy.io.segy.segy import _read_segy
import matplotlib.pyplot as plt
import os
import pandas as pd


def save_to_txt(data, filepath):
    """
    保存数据为TXT格式
    """
    try:
        if data.ndim == 2:
            np.savetxt(filepath, data, fmt='%.6f', delimiter='\t')
        else:
            flat_data = data.flatten()
            np.savetxt(filepath, flat_data, fmt='%.6f', delimiter='\t')
        print(f"保存为TXT格式: {os.path.basename(filepath)}")
    except Exception as e:
        print(f"保存TXT失败: {e}")


def save_to_csv(data, filepath):
    """
    保存数据为CSV格式
    """
    try:
        if data.ndim == 1:
            df = pd.DataFrame({'data': data})
        elif data.ndim == 2:
            df = pd.DataFrame(data)
            df.columns = [f'col_{i}' for i in range(data.shape[1])]
        else:
            flat_data = data.flatten()
            df = pd.DataFrame({'data': flat_data})
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"保存为CSV格式: {os.path.basename(filepath)}")
    except Exception as e:
        print(f"保存CSV失败: {e}")


def read_segy_with_obspy(file_path):
    """
    使用 ObsPy 读取 SEGY 数据，并返回数据矩阵
    """
    print(f"正在读取 SEGY 文件: {os.path.basename(file_path)}")

    try:
        segy = _read_segy(file_path)
        traces_data = []
        for trace in segy.traces:
            traces_data.append(trace.data)
        data = np.array(traces_data)
        print(f"  原始数据形状: {data.shape}")
        data = data.T
        print(f"  转置后数据形状: {data.shape}")
        return data
    except Exception as e:
        print(f"  读取失败: {e}")
        try:
            st = obspy.read(file_path, format='SEGY')
            data = np.array([tr.data for tr in st])
            print(f"  原始数据形状: {data.shape}")
            data = data.T
            print(f"  转置后数据形状: {data.shape}")
            return data
        except Exception as e2:
            print(f"  obspy.read 也失败了: {e2}")
            return None


def save_to_bin(data, file_path):
    """保存为二进制格式"""
    print(f"保存为BIN格式: {os.path.basename(file_path)}")
    data.astype(np.float32).tofile(file_path)


def save_to_mat(data, file_path):
    """保存为MATLAB格式"""
    print(f"保存为MAT格式: {os.path.basename(file_path)}")
    from scipy.io import savemat
    savemat(file_path, {'data': data})


def save_to_npy(data, file_path):
    """保存为NPY格式"""
    print(f"保存为NPY格式: {os.path.basename(file_path)}")
    np.save(file_path, data)


def save_to_dat(data, file_path):
    """保存为DAT文本格式"""
    print(f"保存为DAT格式: {os.path.basename(file_path)}")
    np.savetxt(file_path, data, fmt='%.6f', delimiter='\t')


def get_file_base_name(file_path):
    """
    从文件路径获取基础名称（不含扩展名）
    """
    return os.path.splitext(os.path.basename(file_path))[0]


def process_single_segy(segy_file, data_dir, fig_dir, save_formats=True):
    """
    处理单个SEGY文件 - 只保存数据，不生成单独的图
    """
    print(f"\n{'=' * 60}")
    print(f"处理文件: {os.path.basename(segy_file)}")
    print('=' * 60)

    base_name = get_file_base_name(segy_file)
    data = read_segy_with_obspy(segy_file)

    if data is None or data.size == 0:
        print(f"错误: 读取 {segy_file} 失败")
        return None, base_name

    if save_formats:
        print(f"\n保存 {base_name} 为多种格式...")
        save_to_npy(data, os.path.join(data_dir, f'{base_name}.npy'))
        save_to_bin(data, os.path.join(data_dir, f'{base_name}.bin'))
        save_to_mat(data, os.path.join(data_dir, f'{base_name}.mat'))
        save_to_dat(data, os.path.join(data_dir, f'{base_name}.dat'))
        save_to_txt(data, os.path.join(data_dir, f'{base_name}.txt'))
        save_to_csv(data, os.path.join(data_dir, f'{base_name}.csv'))

    return data, base_name


def load_data_memory_efficient(data_dir, base_name, data_shape):
    """
    内存高效地加载数据，使用内存映射避免同时加载所有数据
    """
    data_dict = {}

    # 加载SEGY数据
    print("  加载 SEGY (内存映射)...")
    try:
        segy_file = os.path.join(data_dir, f'{base_name}.segy')
        if os.path.exists(segy_file):
            segy = _read_segy(segy_file)
            traces_data = []
            for trace in segy.traces:
                traces_data.append(trace.data)
            data_dict['segy'] = np.array(traces_data).T
            print(f"    SEGY: {data_dict['segy'].shape}")
    except Exception as e:
        print(f"    加载SEGY失败: {e}")
        data_dict['segy'] = None

    # 加载其他格式（使用内存映射）
    file_names = {
        'npy': f'{base_name}.npy',
        'bin': f'{base_name}.bin',
        'mat': f'{base_name}.mat',
        'dat': f'{base_name}.dat',
        'txt': f'{base_name}.txt',
        'csv': f'{base_name}.csv'
    }

    for fmt in ['npy', 'bin', 'mat', 'dat', 'txt', 'csv']:
        file_path = os.path.join(data_dir, file_names[fmt])
        if not os.path.exists(file_path):
            data_dict[fmt] = None
            continue
        try:
            if fmt == 'npy':
                # 使用mmap模式加载（内存映射）
                data_dict[fmt] = np.load(file_path, mmap_mode='r')
                print(f"    NPY (内存映射): {data_dict[fmt].shape}")
            elif fmt == 'bin':
                # 使用内存映射
                if data_shape is not None:
                    data_dict[fmt] = np.memmap(file_path, dtype=np.float32, mode='r', shape=data_shape)
                    print(f"    BIN (内存映射): {data_dict[fmt].shape}")
                else:
                    # 如果不知道形状，先读取大小再reshape
                    data = np.fromfile(file_path, dtype=np.float32)
                    # 尝试从npy获取形状
                    if 'npy' in data_dict and data_dict['npy'] is not None:
                        try:
                            data = data.reshape(data_dict['npy'].shape)
                            data_dict[fmt] = data
                            print(f"    BIN: {data_dict[fmt].shape}")
                        except:
                            data_dict[fmt] = data
                            print(f"    BIN (1D): {data.shape}")
                    else:
                        data_dict[fmt] = data
                        print(f"    BIN (1D): {data.shape}")
            elif fmt == 'mat':
                from scipy.io import loadmat
                mat_data = loadmat(file_path)
                data_dict[fmt] = mat_data['data']
                print(f"    MAT: {data_dict[fmt].shape}")
            elif fmt == 'dat':
                # dat文件直接加载（文本文件无法内存映射）
                data_dict[fmt] = np.loadtxt(file_path)
                print(f"    DAT: {data_dict[fmt].shape}")
            elif fmt == 'txt':
                # txt文件直接加载（文本文件无法内存映射）
                data_dict[fmt] = np.loadtxt(file_path)
                print(f"    TXT: {data_dict[fmt].shape}")
            elif fmt == 'csv':
                # csv文件直接加载（文本文件无法内存映射）
                csv_data = pd.read_csv(file_path)
                data_dict[fmt] = csv_data.values
                print(f"    CSV: {data_dict[fmt].shape}")
        except Exception as e:
            print(f"    加载{fmt.upper()}失败: {e}")
            data_dict[fmt] = None

    return data_dict


def load_and_visualize_all_formats(data_dir, fig_dir):
    """
    加载所有格式的数据并绘制对比图（使用内存映射）
    """
    print("\n加载所有数据并绘制对比图...")

    # 获取所有数据集
    data_files = {}
    for file in os.listdir(data_dir):
        if file.endswith(('.npy', '.bin', '.mat', '.dat', '.txt', '.csv')):
            base_name = os.path.splitext(file)[0]
            ext = os.path.splitext(file)[1][1:]
            if base_name not in data_files:
                data_files[base_name] = {}
            data_files[base_name][ext] = os.path.join(data_dir, file)

        if file.lower().endswith('.segy'):
            base_name = os.path.splitext(file)[0]
            if base_name not in data_files:
                data_files[base_name] = {}
            data_files[base_name]['segy'] = os.path.join(data_dir, file)

    if not data_files:
        print("未找到任何数据文件")
        return

    print(f"找到 {len(data_files)} 个数据集:")
    for name in data_files.keys():
        formats = list(data_files[name].keys())
        print(f"  - {name}: {', '.join([f.upper() for f in formats])}")

    for base_name, formats in data_files.items():
        print(f"\n生成 {base_name} 的对比图...")
        visualize_format_comparison(base_name, formats, fig_dir)


def visualize_format_comparison(base_name, formats, fig_dir):
    """
    为单个数据集生成多格式对比图（使用内存映射）
    """
    format_labels = {
        'segy': 'SEGY',
        'npy': 'NPY (mmap)',
        'bin': 'BIN (mmap)',
        'mat': 'MAT',
        'dat': 'DAT',
        'txt': 'TXT',
        'csv': 'CSV'
    }

    # 先获取数据形状（从npy或segy）
    data_shape = None
    if 'npy' in formats:
        try:
            # 只读取形状，不加载数据
            with open(formats['npy'], 'rb') as f:
                # 读取npy文件头获取形状
                import struct
                # 简单方法：用np.load查看形状但不加载全部
                temp = np.load(formats['npy'], mmap_mode='r')
                data_shape = temp.shape
                del temp
        except:
            pass

    if data_shape is None and 'segy' in formats:
        try:
            segy = _read_segy(formats['segy'])
            data_shape = (len(segy.traces[0].data), len(segy.traces))
        except:
            pass

    # 使用内存高效加载
    data_dict = load_data_memory_efficient(os.path.dirname(list(formats.values())[0]),
                                           base_name, data_shape)

    format_order = ['segy', 'npy', 'bin', 'mat', 'dat', 'txt', 'csv']
    available_formats = [f for f in format_order if f in data_dict and data_dict[f] is not None]
    n_formats = len(available_formats)

    if n_formats == 0:
        print(f"  {base_name}: 没有可用的数据")
        return

    print(f"  成功加载 {n_formats} 种格式 (使用内存映射)")

    # 计算行列数
    n_cols = min(4, n_formats)
    n_rows = (n_formats + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    plot_idx = 0
    for fmt in available_formats:
        data = data_dict[fmt]

        # 处理数据维度
        if data.ndim == 1:
            if len(data) > 0:
                sqrt_n = int(np.sqrt(len(data)))
                if sqrt_n * sqrt_n == len(data):
                    data = data.reshape(sqrt_n, sqrt_n)
                else:
                    axes[plot_idx].plot(data, linewidth=1.5)
                    axes[plot_idx].set_title(f'{format_labels.get(fmt, fmt.upper())}\n(1D, n={len(data)})',
                                             fontsize=12, fontweight='bold')
                    axes[plot_idx].set_xlabel('Index', fontsize=10)
                    axes[plot_idx].set_ylabel('Value', fontsize=10)
                    axes[plot_idx].grid(True, alpha=0.3)
                    plot_idx += 1
                    continue
        elif data.ndim > 2:
            data = data[0] if data.shape[0] > 0 else data

        # 显示2D数据
        im = axes[plot_idx].imshow(data, aspect='auto', cmap='jet', interpolation='bilinear')
        axes[plot_idx].set_title(f'{format_labels.get(fmt, fmt.upper())}\n{data.shape}',
                                 fontsize=13, fontweight='bold')
        axes[plot_idx].set_xlabel('Trace Number', fontsize=10)
        axes[plot_idx].set_ylabel('Sample Point', fontsize=10)
        axes[plot_idx].tick_params(labelsize=9)

        cbar = plt.colorbar(im, ax=axes[plot_idx])
        cbar.ax.tick_params(labelsize=9)
        cbar.set_label('Amplitude', fontsize=9)

        # 显示数据信息
        info_text = f'Min: {data.min():.2f}\nMax: {data.max():.2f}'
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

    plt.suptitle(f'{base_name} - Multi-Format Comparison (Memory Mapped)',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    fig_path = os.path.join(fig_dir, f'{base_name}_format_comparison.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ 对比图已保存: {os.path.basename(fig_path)}")
    plt.close()


def batch_process_segy_files(data_dir, segy_files=None, auto_discover=True):
    """
    批量处理SEGY文件
    """
    fig_dir = os.path.join(data_dir, 'figs')
    os.makedirs(fig_dir, exist_ok=True)

    if segy_files is None and auto_discover:
        segy_files = []
        for file in os.listdir(data_dir):
            if file.lower().endswith('.segy'):
                segy_files.append(os.path.join(data_dir, file))

    if not segy_files:
        print("错误: 没有找到任何SEGY文件")
        return

    print(f"\n找到 {len(segy_files)} 个SEGY文件:")
    for f in segy_files:
        print(f"  - {os.path.basename(f)}")

    processed_data = {}
    for segy_file in segy_files:
        data, base_name = process_single_segy(segy_file, data_dir, fig_dir,
                                              save_formats=True)
        if data is not None:
            processed_data[base_name] = data

    if processed_data:
        load_and_visualize_all_formats(data_dir, fig_dir)

    return processed_data


def main():
    data_dir = 'data'

    print("开始处理所有SEGY文件...")
    processed_data = batch_process_segy_files(data_dir, auto_discover=True)

    print("\n" + "=" * 60)
    print("所有处理完成！")
    print(f"数据保存在: {data_dir}")
    print(f"对比图保存在: {os.path.join(data_dir, 'figs')}")
    print("=" * 60)


if __name__ == "__main__":
    main()