import numpy as np
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import loadmat, savemat
from scipy.ndimage import zoom
from obspy.io.segy.segy import _read_segy


# ==================== 保存函数 ====================
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


# ==================== 加载函数 ====================
def load_from_npy(file_path):
    """从NPY加载数据"""
    return np.load(file_path)


def load_from_bin(file_path, shape=None):
    """从BIN加载数据"""
    try:
        data = np.fromfile(file_path, dtype=np.float32)
        if shape is not None:
            return data.reshape(shape)
        return data
    except Exception as e:
        print(f"    加载BIN失败: {e}")
        return None


def load_from_mat(file_path):
    """从MAT加载数据"""
    try:
        data = loadmat(file_path)
        for key in data.keys():
            if not key.startswith('__'):
                return data[key]
        return None
    except Exception as e:
        print(f"    加载MAT失败: {e}")
        return None


def load_from_dat(file_path):
    """从DAT加载数据"""
    try:
        return np.loadtxt(file_path)
    except Exception as e:
        print(f"    加载DAT失败: {e}")
        return None


def load_from_txt(file_path):
    """从TXT加载数据"""
    try:
        return np.loadtxt(file_path)
    except Exception as e:
        print(f"    加载TXT失败: {e}")
        return None


def load_from_csv(file_path):
    """从CSV加载数据"""
    try:
        df = pd.read_csv(file_path, header=None)
        return df.values
    except Exception as e:
        print(f"    加载CSV失败: {e}")
        return None


# 格式标签
FORMAT_LABELS = {
    'npy': 'NPY',
    'bin': 'BIN',
    'mat': 'MAT',
    'dat': 'DAT',
    'txt': 'TXT',
    'csv': 'CSV'
}

LOAD_FUNCTIONS = {
    'npy': load_from_npy,
    'bin': load_from_bin,
    'mat': load_from_mat,
    'dat': load_from_dat,
    'txt': load_from_txt,
    'csv': load_from_csv
}

SAVE_FUNCTIONS = {
    'npy': save_to_npy,
    'bin': save_to_bin,
    'mat': save_to_mat,
    'dat': save_to_dat,
    'txt': save_to_txt,
    'csv': save_to_csv
}


# ==================== SEGY读取函数 ====================
def read_segy_with_obspy(file_path):
    """使用 ObsPy 读取 SEGY 数据"""
    print(f"  读取 SEGY 文件: {os.path.basename(file_path)}")
    try:
        segy = _read_segy(file_path)
        traces_data = []
        for trace in segy.traces:
            traces_data.append(trace.data)
        data = np.array(traces_data)
        data = data.T
        print(f"    数据形状: {data.shape}")
        return data
    except Exception as e:
        print(f"    读取失败: {e}")
        return None


# ==================== 格式转换函数 ====================
def convert_all_formats(source_data, output_dir, base_name, source_format=None):
    """
    将源数据转换为所有格式
    source_format: 源格式名称，用于命名
    """
    formats = ['npy', 'bin', 'mat', 'dat', 'txt', 'csv']
    converted_data = {}

    for target_fmt in formats:
        if target_fmt == source_format:
            continue

        # 生成文件名
        if source_format:
            conv_name = f'{base_name}_from_{source_format}_to_{target_fmt}'
        else:
            conv_name = f'{base_name}_to_{target_fmt}'

        output_path = os.path.join(output_dir, f'{conv_name}.{target_fmt}')

        try:
            if target_fmt == 'bin':
                save_to_bin(source_data, output_path)
            else:
                SAVE_FUNCTIONS[target_fmt](source_data, output_path)

            key = f'{source_format}->{target_fmt}' if source_format else f'orig->{target_fmt}'
            converted_data[key] = {
                'data': source_data.copy(),
                'path': output_path,
                'source': source_format or 'original',
                'target': target_fmt
            }
        except Exception as e:
            print(f"    ✗ 保存失败: {e}")

    return converted_data


# ==================== 绘图函数 ====================
def plot_comparison(data_dict, fig_dir, base_name, title_suffix=""):
    """
    绘制多格式对比图
    """
    if not data_dict:
        print("  没有可用数据绘制对比图")
        return None

    n_datasets = len(data_dict)
    n_cols = min(4, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    plot_idx = 0
    for key, info in data_dict.items():
        data = info['data']
        source = info.get('source', 'orig')
        target = info.get('target', '')

        if source == 'orig':
            title = f'Original Data\n{data.shape}'
        else:
            title = f'{FORMAT_LABELS.get(source, source)} → {FORMAT_LABELS.get(target, target)}\n{data.shape}'

        # 处理1D数据
        if data.ndim == 1:
            axes[plot_idx].plot(data, linewidth=1.5)
            axes[plot_idx].set_title(title, fontsize=11, fontweight='bold')
            axes[plot_idx].set_xlabel('Index', fontsize=9)
            axes[plot_idx].set_ylabel('Value', fontsize=9)
            axes[plot_idx].grid(True, alpha=0.3)
        else:
            # 显示2D数据
            im = axes[plot_idx].imshow(data, aspect='auto', cmap='jet',
                                       interpolation='bilinear')
            axes[plot_idx].set_title(title, fontsize=11, fontweight='bold')
            axes[plot_idx].set_xlabel('X (Trace)', fontsize=9)
            axes[plot_idx].set_ylabel('Y (Sample)', fontsize=9)
            axes[plot_idx].tick_params(labelsize=8)

            cbar = plt.colorbar(im, ax=axes[plot_idx])
            cbar.ax.tick_params(labelsize=8)
            cbar.set_label('Value', fontsize=8)

        # 显示统计信息
        info_text = f'Min: {data.min():.2f}\nMax: {data.max():.2f}'
        axes[plot_idx].text(0.02, 0.98, info_text,
                            transform=axes[plot_idx].transAxes,
                            fontsize=7, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        plot_idx += 1

    # 隐藏多余的子图
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'Data Format Comparison{title_suffix}\n{base_name}',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    fig_path = os.path.join(fig_dir, f'{base_name}_comparison.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ 对比图已保存: {fig_path}")
    plt.close()

    return fig_path


# ==================== 主程序 ====================
def main():
    # 配置
    input_dir = 'data'
    output_dir = 'output_data'
    fig_dir = os.path.join(output_dir, 'figs')

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    start_time = time.time()

    print("=" * 80)
    print("SEGY数据处理与格式转换工具")
    print("=" * 80)

    # ====== 步骤1: 查找SEGY文件 ======
    print("\n[1] 查找SEGY文件...")
    segy_files = []
    for file in os.listdir(input_dir):
        if file.lower().endswith('.segy'):
            segy_files.append(os.path.join(input_dir, file))

    if not segy_files:
        print("  未找到SEGY文件，尝试查找MAT文件...")
        mat_files = []
        for file in os.listdir(input_dir):
            if file.lower().endswith('.mat'):
                mat_files.append(os.path.join(input_dir, file))

        if mat_files:
            print(f"  找到MAT文件: {os.path.basename(mat_files[0])}")
            process_mat_file(mat_files[0], output_dir, fig_dir)
        else:
            print("  错误: 未找到任何SEGY或MAT文件")
            return
    else:
        print(f"  找到 {len(segy_files)} 个SEGY文件:")
        for f in segy_files:
            print(f"    - {os.path.basename(f)}")

        # 处理第一个SEGY文件
        process_segy_file(segy_files[0], output_dir, fig_dir)

    # ====== 步骤4: 总结 ======
    print("\n" + "=" * 80)
    print("处理完成！")
    print("=" * 80)
    print(f"数据目录: {output_dir}")
    print(f"图片目录: {fig_dir}")
    print(f"总耗时: {time.time() - start_time:.2f}秒")
    print("=" * 80)


def process_segy_file(segy_file, output_dir, fig_dir):
    """处理SEGY文件"""
    base_name = os.path.splitext(os.path.basename(segy_file))[0]

    print(f"\n[2] 处理SEGY文件: {base_name}")

    # 读取SEGY
    data = read_segy_with_obspy(segy_file)
    if data is None:
        print("  错误: 读取SEGY失败")
        return

    # 保存原始数据为NPY（作为基准）
    print(f"\n  保存原始数据为NPY格式...")
    np.save(os.path.join(output_dir, f'{base_name}_original.npy'), data)

    # 转换为所有格式
    print(f"\n[3] 转换为多种格式...")
    all_data = {}

    # 添加原始数据
    all_data['original'] = {
        'data': data,
        'source': 'orig',
        'target': ''
    }

    # 转换为其他格式
    converted = convert_all_formats(data, output_dir, base_name, 'segy')
    all_data.update(converted)

    # 绘制对比图
    print(f"\n[4] 生成对比图...")
    plot_comparison(all_data, fig_dir, base_name, " (SEGY Source)")


def process_mat_file(mat_file, output_dir, fig_dir):
    """处理MAT文件（10倍重采样）"""
    base_name = os.path.splitext(os.path.basename(mat_file))[0]

    print(f"\n[2] 处理MAT文件: {base_name}")

    # 加载MAT数据
    print("  加载MAT数据...")
    try:
        mat_data = loadmat(mat_file)
        data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
        if data_keys:
            data_key = data_keys[0]
            original_data = mat_data[data_key]
            print(f"    变量名: '{data_key}', 形状: {original_data.shape}")
        else:
            raise ValueError("未找到数据变量")

        if original_data.ndim > 2:
            original_data = original_data.squeeze()

        print(f"    数据范围: [{original_data.min():.2f}, {original_data.max():.2f}]")
    except Exception as e:
        print(f"  加载失败: {e}")
        return

    # 10倍重采样
    print("\n[3] 10倍重采样...")
    original_shape = original_data.shape

    if len(original_shape) == 2:
        target_shape = (int(round(original_shape[0] / 10)),
                        int(round(original_shape[1] / 10)))
        zoom_factors = (target_shape[0] / original_shape[0],
                        target_shape[1] / original_shape[1])
        resampled_data = zoom(original_data, zoom_factors, order=1)

        print(f"    原始形状: {original_shape}")
        print(f"    目标形状: {target_shape}")
        print(f"    缩放因子: {zoom_factors[0]:.4f}, {zoom_factors[1]:.4f}")
        print(f"    重采样后范围: [{resampled_data.min():.2f}, {resampled_data.max():.2f}]")
    else:
        print(f"    警告: 数据不是2D ({original_shape})，跳过重采样")
        resampled_data = original_data

    # 保存重采样后的数据
    resampled_base = f'{base_name}_resampled_10x'
    print(f"\n[4] 保存重采样数据为多种格式...")

    all_data = {}

    # 保存原始重采样数据（基准）
    np.save(os.path.join(output_dir, f'{resampled_base}_original.npy'), resampled_data)

    # 转换为所有格式
    converted = convert_all_formats(resampled_data, output_dir, resampled_base, 'mat')
    all_data.update(converted)

    # 绘制对比图
    print(f"\n[5] 生成对比图...")
    plot_comparison(all_data, fig_dir, resampled_base, " (10x Downsampled)")


if __name__ == "__main__":
    main()