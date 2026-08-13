#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SEGY数据读取与多格式转换工具
功能：
1. 读取SEGY文件并保存为多种格式 (NPY, BIN, MAT, DAT, TXT, CSV)
2. 对MAT数据进行重采样并保存为多种格式
3. 格式互转工具
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
from scipy.io import loadmat, savemat
from scipy.ndimage import zoom


# ============================================================
# 第一部分: 公共工具函数
# ============================================================

def get_file_base_name(file_path):
    """从文件路径获取基础名称（不含扩展名）"""
    return os.path.splitext(os.path.basename(file_path))[0]


# ============================================================
# 第二部分: SEGY读取与保存 (原始程序1)
# ============================================================

def read_segy_with_obspy(file_path):
    """
    使用 ObsPy 读取 SEGY 数据，并返回数据矩阵
    """
    print(f"正在读取 SEGY 文件: {os.path.basename(file_path)}")

    try:
        from obspy.io.segy.segy import _read_segy
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
        print(f"  _read_segy 读取失败: {e}")
        try:
            import obspy
            st = obspy.read(file_path, format='SEGY')
            data = np.array([tr.data for tr in st])
            print(f"  原始数据形状: {data.shape}")
            data = data.T
            print(f"  转置后数据形状: {data.shape}")
            return data
        except Exception as e2:
            print(f"  obspy.read 也失败了: {e2}")
            return None


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
    if data.ndim == 1:
        df = pd.DataFrame({'data': data})
    else:
        df = pd.DataFrame(data)
    df.to_csv(file_path, index=False, header=False)
    print(f"  ✓ CSV: {os.path.basename(file_path)}")


# 保存函数映射
SAVE_FUNCTIONS = {
    'npy': save_to_npy,
    'bin': save_to_bin,
    'mat': save_to_mat,
    'dat': save_to_dat,
    'txt': save_to_txt,
    'csv': save_to_csv
}

FORMAT_LABELS = {
    'npy': 'NPY',
    'bin': 'BIN',
    'mat': 'MAT',
    'dat': 'DAT',
    'txt': 'TXT',
    'csv': 'CSV'
}


def save_all_formats(data, data_dir, base_name):
    """保存数据为所有格式"""
    formats = ['npy', 'bin', 'mat', 'dat', 'txt', 'csv']
    for fmt in formats:
        file_path = os.path.join(data_dir, f'{base_name}.{fmt}')
        SAVE_FUNCTIONS[fmt](data, file_path)


def process_single_segy(segy_file, data_dir):
    """
    处理单个SEGY文件 - 读取并保存为多种格式
    """
    print(f"\n{'=' * 60}")
    print(f"处理 SEGY 文件: {os.path.basename(segy_file)}")
    print('=' * 60)

    base_name = get_file_base_name(segy_file)
    data = read_segy_with_obspy(segy_file)

    if data is None or data.size == 0:
        print(f"错误: 读取 {segy_file} 失败")
        return None, base_name

    print(f"\n保存 {base_name} 为多种格式...")
    save_all_formats(data, data_dir, base_name)

    return data, base_name


def batch_process_segy_files(data_dir):
    """
    批量处理SEGY文件
    """
    # 查找所有SEGY文件
    segy_files = []
    for file in os.listdir(data_dir):
        if file.lower().endswith('.segy'):
            segy_files.append(os.path.join(data_dir, file))

    if not segy_files:
        print("警告: 没有找到任何SEGY文件")
        return {}

    print(f"\n找到 {len(segy_files)} 个SEGY文件:")
    for f in segy_files:
        print(f"  - {os.path.basename(f)}")

    processed_data = {}
    for segy_file in segy_files:
        data, base_name = process_single_segy(segy_file, data_dir)
        if data is not None:
            processed_data[base_name] = data

    return processed_data


# ============================================================
# 第三部分: MAT数据重采样与保存 (原始程序2)
# ============================================================

def process_mat_resample(mat_file, data_dir):
    """
    处理MAT文件：重采样并保存为多种格式
    """
    print("\n" + "=" * 60)
    print("处理 MAT 文件重采样")
    print("=" * 60)

    base_name = get_file_base_name(mat_file)
    output_base = f'{base_name}_resampled_10x'

    # 1. 加载原始数据
    print("\n[1] 加载原始数据...")
    start_time = time.time()

    try:
        mat_data = loadmat(mat_file)
        data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
        if data_keys:
            data_key = data_keys[0]
            original_data = mat_data[data_key]
            print(f"  从MAT文件加载: 变量名 '{data_key}', 形状 {original_data.shape}")
        else:
            raise ValueError("未找到数据变量")

        if original_data.ndim > 2:
            original_data = original_data.squeeze()

        print(f"  原始数据形状: {original_data.shape}")
        print(f"  原始数据范围: [{original_data.min():.2f}, {original_data.max():.2f}]")

    except Exception as e:
        print(f"  加载失败: {e}")
        return None, output_base

    # 2. 10倍重采样
    print("\n[2] 10倍重采样...")
    original_shape = original_data.shape
    target_shape = (int(round(original_shape[0] / 10)), int(round(original_shape[1] / 10)))
    print(f"  原始形状: {original_shape}")
    print(f"  目标形状: {target_shape}")

    zoom_factors = (target_shape[0] / original_shape[0], target_shape[1] / original_shape[1])
    resampled_data = zoom(original_data, zoom_factors, order=1)

    print(f"  重采样后形状: {resampled_data.shape}")
    print(f"  重采样后范围: [{resampled_data.min():.2f}, {resampled_data.max():.2f}]")
    print(f"  重采样耗时: {time.time() - start_time:.2f}秒")

    # 3. 保存为多种格式
    print("\n[3] 保存为多种格式...")
    save_all_formats(resampled_data, data_dir, output_base)

    return resampled_data, output_base


# ============================================================
# 第四部分: 格式互转工具 (原始程序3)
# ============================================================

def load_from_npy(file_path):
    """从NPY加载数据"""
    return np.load(file_path)


def load_from_bin(file_path, shape=None):
    """从BIN加载数据"""
    try:
        data = np.fromfile(file_path, dtype=np.float32)
        if shape is not None:
            try:
                return data.reshape(shape)
            except Exception:
                print(f"    警告: 无法重塑为 {shape}，返回1D数据")
                return data
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


LOAD_FUNCTIONS = {
    'npy': load_from_npy,
    'bin': load_from_bin,
    'mat': load_from_mat,
    'dat': load_from_dat,
    'txt': load_from_txt,
    'csv': load_from_csv
}


def convert_formats(data_dir, output_dir):
    """
    格式互转：从每种格式加载数据，转换成其他格式
    """
    print("\n" + "=" * 60)
    print("数据格式互转工具")
    print("=" * 60)

    formats = ['npy', 'bin', 'mat', 'dat', 'txt', 'csv']

    # 查找所有可用的源文件
    print("\n[1] 查找源数据文件...")
    source_files = {}
    base_names = set()

    for file in os.listdir(data_dir):
        for fmt in formats:
            if file.endswith(f'.{fmt}'):
                base = file[:-len(fmt) - 1]
                base_names.add(base)

    if not base_names:
        print("  未找到任何数据文件")
        return {}

    print(f"  找到 {len(base_names)} 个数据集")

    all_converted_data = {}

    for base_name in base_names:
        print(f"\n处理数据集: {base_name}")

        # 收集该数据集的所有格式文件
        source_files = {}
        for fmt in formats:
            file_path = os.path.join(data_dir, f'{base_name}.{fmt}')
            if os.path.exists(file_path):
                source_files[fmt] = file_path
                size = os.path.getsize(file_path) / (1024 * 1024)
                print(f"  ✓ {fmt.upper()}: {os.path.basename(file_path)} ({size:.2f} MB)")

        if len(source_files) < 2:
            print(f"  跳过: 只有 {len(source_files)} 种格式，需要至少2种")
            continue

        # 加载所有源数据
        source_data_cache = {}
        for fmt, path in source_files.items():
            print(f"  加载 {fmt.upper()}...")
            try:
                if fmt == 'bin':
                    # 尝试从其他格式获取形状
                    shape = None
                    for other_fmt in source_files:
                        if other_fmt != 'bin' and other_fmt in source_data_cache:
                            shape = source_data_cache[other_fmt].shape
                            break
                    if shape is not None:
                        data = load_from_bin(path, shape)
                    else:
                        # 尝试从npy文件读取形状
                        npy_path = os.path.join(data_dir, f'{base_name}.npy')
                        if os.path.exists(npy_path):
                            temp = np.load(npy_path, mmap_mode='r')
                            shape = temp.shape
                            data = load_from_bin(path, shape)
                        else:
                            data = load_from_bin(path)
                else:
                    data = LOAD_FUNCTIONS[fmt](path)

                if data is not None and data.ndim == 2:
                    source_data_cache[fmt] = data
                    print(f"    成功: {data.shape}")
                else:
                    print(f"    跳过: 数据维度 {data.ndim if data is not None else 'None'}")

            except Exception as e:
                print(f"    失败: {e}")

        if len(source_data_cache) < 2:
            print(f"  跳过: 成功加载 {len(source_data_cache)} 种格式，需要至少2种")
            continue

        # 执行转换
        print(f"  开始转换...")
        for source_fmt, source_data in source_data_cache.items():
            target_formats = [f for f in formats if f != source_fmt]
            for target_fmt in target_formats:
                converted_name = f'{base_name}_from_{source_fmt}_to_{target_fmt}'
                output_path = os.path.join(output_dir, f'{converted_name}.{target_fmt}')

                try:
                    SAVE_FUNCTIONS[target_fmt](source_data, output_path)
                    key = f'{source_fmt}->{target_fmt}'
                    all_converted_data[key] = source_data
                    print(f"    ✓ {FORMAT_LABELS[source_fmt]} → {FORMAT_LABELS[target_fmt]}")
                except Exception as e:
                    print(f"    ✗ {FORMAT_LABELS[source_fmt]} → {FORMAT_LABELS[target_fmt]} 失败: {e}")

    return all_converted_data


# ============================================================
# 第五部分: 可视化工具
# ============================================================

def visualize_all_data(data_dir, fig_dir, base_name, data_dict=None):
    """
    可视化所有格式的数据对比
    """
    formats = ['npy', 'bin', 'mat', 'dat', 'txt', 'csv']

    # 如果提供了数据字典，直接使用
    if data_dict is not None:
        available_formats = [fmt for fmt, data in data_dict.items() if data is not None]
        plot_data = data_dict
    else:
        # 否则从文件加载
        plot_data = {}
        for fmt in formats:
            file_path = os.path.join(data_dir, f'{base_name}.{fmt}')
            if os.path.exists(file_path):
                try:
                    if fmt == 'npy':
                        plot_data[fmt] = np.load(file_path)
                    elif fmt == 'bin':
                        # 尝试从npy获取形状
                        npy_path = os.path.join(data_dir, f'{base_name}.npy')
                        if os.path.exists(npy_path):
                            temp = np.load(npy_path, mmap_mode='r')
                            plot_data[fmt] = np.fromfile(file_path, dtype=np.float32).reshape(temp.shape)
                        else:
                            plot_data[fmt] = np.fromfile(file_path, dtype=np.float32)
                    elif fmt == 'mat':
                        mat_data = loadmat(file_path)
                        for key in mat_data.keys():
                            if not key.startswith('__'):
                                plot_data[fmt] = mat_data[key]
                                break
                    elif fmt == 'dat':
                        plot_data[fmt] = np.loadtxt(file_path)
                    elif fmt == 'txt':
                        plot_data[fmt] = np.loadtxt(file_path)
                    elif fmt == 'csv':
                        df = pd.read_csv(file_path, header=None)
                        plot_data[fmt] = df.values
                except Exception as e:
                    print(f"  加载 {fmt.upper()} 失败: {e}")

        available_formats = [fmt for fmt in formats if fmt in plot_data and plot_data[fmt] is not None]

    if not available_formats:
        print("  没有可用的数据")
        return

    print(f"  可用格式: {', '.join([f.upper() for f in available_formats])}")

    # 过滤2D数据
    valid_data = {}
    for fmt in available_formats:
        data = plot_data[fmt]
        if data is not None and data.ndim == 2:
            valid_data[fmt] = data

    if not valid_data:
        print("  没有2D数据可用于可视化")
        return

    n_datasets = len(valid_data)
    n_cols = min(3, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))

    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    plot_idx = 0
    for fmt, data in valid_data.items():
        im = axes[plot_idx].imshow(data, aspect='auto', cmap='jet', interpolation='bilinear')
        axes[plot_idx].set_title(f'{FORMAT_LABELS.get(fmt, fmt.upper())}\n{data.shape}',
                                 fontsize=12, fontweight='bold')
        axes[plot_idx].set_xlabel('Trace Number', fontsize=9)
        axes[plot_idx].set_ylabel('Sample Point', fontsize=9)
        axes[plot_idx].tick_params(labelsize=8)

        cbar = plt.colorbar(im, ax=axes[plot_idx])
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label('Amplitude', fontsize=8)

        info_text = f'Min: {data.min():.2f}\nMax: {data.max():.2f}'
        axes[plot_idx].text(0.02, 0.98, info_text,
                            transform=axes[plot_idx].transAxes,
                            fontsize=7, verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        plot_idx += 1

    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'{base_name} - Multi-Format Comparison',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    fig_path = os.path.join(fig_dir, f'{base_name}_comparison.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ 对比图已保存: {os.path.basename(fig_path)}")
    plt.close()


def visualize_conversion_results(output_dir, fig_dir, converted_data):
    """
    可视化格式转换结果
    """
    if not converted_data:
        print("  没有转换数据")
        return

    n_datasets = len(converted_data)
    n_cols = min(4, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    plot_idx = 0
    for key, data in converted_data.items():
        source, target = key.split('->')
        title = f'{FORMAT_LABELS[source]} → {FORMAT_LABELS[target]}'

        im = axes[plot_idx].imshow(data, aspect='auto', cmap='jet', interpolation='bilinear')
        axes[plot_idx].set_title(f'{title}\n{data.shape}', fontsize=10, fontweight='bold')
        axes[plot_idx].set_xlabel('X', fontsize=8)
        axes[plot_idx].set_ylabel('Y', fontsize=8)
        axes[plot_idx].tick_params(labelsize=7)

        plt.colorbar(im, ax=axes[plot_idx])
        plot_idx += 1

    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Format Conversion Results', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    fig_path = os.path.join(fig_dir, 'conversion_results.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ 转换结果图已保存: {os.path.basename(fig_path)}")
    plt.close()


# ============================================================
# 第六部分: 主程序
# ============================================================

def main():
    # 配置目录
    data_dir = 'data'  # 输入数据目录
    output_dir = 'output_data'  # 输出数据目录
    fig_dir = 'figs'  # 图片目录

    # 创建目录
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    print("=" * 70)
    print("SEGY/MAT 数据处理与格式转换工具")
    print("=" * 70)
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")
    print(f"图片目录: {fig_dir}")

    all_data = {}
    start_time = time.time()

    # ========================================
    # 步骤1: 处理SEGY文件
    # ========================================
    print("\n" + "=" * 70)
    print("步骤 1: 处理 SEGY 文件")
    print("=" * 70)

    segy_data = batch_process_segy_files(data_dir)
    for base_name, data in segy_data.items():
        all_data[base_name] = data
        # 生成对比图
        visualize_all_data(output_dir, fig_dir, base_name, {base_name: data})

    # ========================================
    # 步骤2: 处理MAT文件重采样
    # ========================================
    print("\n" + "=" * 70)
    print("步骤 2: 处理 MAT 文件 (重采样)")
    print("=" * 70)

    mat_files = []
    for file in os.listdir(data_dir):
        if file.lower().endswith('.mat'):
            mat_files.append(os.path.join(data_dir, file))

    if mat_files:
        for mat_file in mat_files:
            resampled_data, output_base = process_mat_resample(mat_file, output_dir)
            if resampled_data is not None:
                all_data[output_base] = resampled_data
                # 生成对比图
                visualize_all_data(output_dir, fig_dir, output_base, {output_base: resampled_data})
    else:
        print("未找到MAT文件，跳过此步骤")

    # ========================================
    # 步骤3: 格式互转
    # ========================================
    print("\n" + "=" * 70)
    print("步骤 3: 格式互转")
    print("=" * 70)

    converted_data = convert_formats(output_dir, output_dir)
    if converted_data:
        visualize_conversion_results(output_dir, fig_dir, converted_data)

    # ========================================
    # 总结
    # ========================================
    print("\n" + "=" * 70)
    print("所有处理完成！")
    print("=" * 70)
    print(f"总耗时: {time.time() - start_time:.2f} 秒")
    print(f"数据保存在: {output_dir}")
    print(f"图片保存在: {fig_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()