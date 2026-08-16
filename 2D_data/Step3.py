import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import os
import pandas as pd
import time


def load_from_npy(file_path):
    """从NPY加载数据"""
    return np.load(file_path)


def load_from_bin(file_path, shape=None):
    """从BIN加载数据 - 需要手动指定形状"""
    try:
        # 读取所有数据
        data = np.fromfile(file_path, dtype=np.float32)

        # 如果指定了形状，尝试reshape
        if shape is not None:
            try:
                return data.reshape(shape)
            except Exception as e:
                print(f"    错误: 无法重塑为 {shape}，数据长度 {len(data)} 不匹配")
                return None
        else:
            # 如果没有指定形状，提示用户输入
            print(f"    警告: 未指定形状，数据长度: {len(data)}")
            return data
    except Exception as e:
        print(f"    加载BIN失败: {e}")
        return None


def load_from_mat(file_path):
    """从MAT加载数据"""
    try:
        data = loadmat(file_path)
        # 查找数据变量
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


def save_to_npy(data, file_path):
    """保存为NPY格式"""
    np.save(file_path, data)


def save_to_bin(data, file_path):
    """保存为二进制格式"""
    data.astype(np.float32).tofile(file_path)


def save_to_mat(data, file_path):
    """保存为MATLAB格式"""
    savemat(file_path, {'data': data})


def save_to_dat(data, file_path):
    """保存为DAT文本格式"""
    np.savetxt(file_path, data, fmt='%.6f', delimiter='\t')


def save_to_txt(data, file_path):
    """保存为TXT格式"""
    np.savetxt(file_path, data, fmt='%.6f', delimiter='\t')


def save_to_csv(data, file_path):
    """保存为CSV格式"""
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False, header=False)


# 加载函数映射
LOAD_FUNCTIONS = {
    'npy': load_from_npy,
    'bin': load_from_bin,
    'mat': load_from_mat,
    'dat': load_from_dat,
    'txt': load_from_txt,
    'csv': load_from_csv
}

# 保存函数映射
SAVE_FUNCTIONS = {
    'npy': save_to_npy,
    'bin': save_to_bin,
    'mat': save_to_mat,
    'dat': save_to_dat,
    'txt': save_to_txt,
    'csv': save_to_csv
}

# 格式标签
FORMAT_LABELS = {
    'npy': 'NPY',
    'bin': 'BIN',
    'mat': 'MAT',
    'dat': 'DAT',
    'txt': 'TXT',
    'csv': 'CSV'
}


def get_user_input_shape():
    """获取用户输入的维度"""
    while True:
        try:
            user_input = input("请输入BIN数据的维度 (例如: 280,1360): ")
            if user_input.lower() == 'skip':
                return None
            parts = user_input.split(',')
            if len(parts) == 2:
                rows = int(parts[0].strip())
                cols = int(parts[1].strip())
                if rows > 0 and cols > 0:
                    return (rows, cols)
                else:
                    print("  维度必须为正整数")
            else:
                print("  请输入两个正整数，用逗号分隔")
        except ValueError:
            print("  输入格式错误，请输入两个正整数，用逗号分隔")
        except KeyboardInterrupt:
            print("\n  用户取消输入")
            return None


def main():
    # 配置
    data_dir = 'output_data'
    output_dir = 'converted_data'
    fig_dir = os.path.join(output_dir, 'figs')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(fig_dir, exist_ok=True)

    base_name = 'MODEL_DENSITY_1.25m_resampled_10x'

    print("=" * 80)
    print("数据格式互转工具")
    print("=" * 80)

    # 1. 定义所有格式
    formats = ['npy', 'bin', 'mat', 'dat', 'txt', 'csv']

    # 2. 检查原始文件是否存在
    print("\n[1] 检查原始数据文件...")
    source_files = {}
    for fmt in formats:
        file_path = os.path.join(data_dir, f'{base_name}.{fmt}')
        if os.path.exists(file_path):
            source_files[fmt] = file_path
            file_size = os.path.getsize(file_path) / (1024 * 1024)
            print(f"  ✓ {fmt.upper()}: {os.path.basename(file_path)} ({file_size:.2f} MB)")
        else:
            print(f"  ✗ {fmt.upper()}: 文件不存在")

    if not source_files:
        print("错误: 没有找到任何原始数据文件")
        return

    print(f"\n找到 {len(source_files)} 种格式的数据")

    # 3. 设置BIN文件的维度
    print("\n[2] 设置BIN文件维度...")
    bin_shape = None

    if 'bin' in source_files:
        print("  检测到BIN文件，需要指定数据维度")
        print("  提示: 根据之前的信息，数据形状为 (280, 1360)")
        bin_shape = get_user_input_shape()

        if bin_shape is None:
            print("  警告: 未指定BIN维度，将跳过BIN格式的转换")
            # 从source_files中移除BIN，但保留文件用于其他格式的转换
            # 实际上，我们可以继续，但BIN加载会失败
        else:
            print(f"  使用维度: {bin_shape}")
    else:
        print("  未检测到BIN文件，无需设置维度")

    # 4. 从每种格式加载数据，然后转换成其他格式
    print("\n[3] 格式转换...")
    start_time = time.time()

    all_converted_data = {}  # 存储所有转换后的数据
    conversion_count = 0

    # 先加载所有源数据到内存（避免重复加载）
    source_data_cache = {}

    for source_fmt, source_path in source_files.items():
        print(f"\n  从 {FORMAT_LABELS[source_fmt]} 加载数据...")

        # 加载源数据
        try:
            if source_fmt == 'bin':
                # 对于BIN格式，使用用户指定的维度
                if bin_shape is None:
                    print(f"    跳过 {source_fmt.upper()} (未指定维度)")
                    continue
                source_data = load_from_bin(source_path, bin_shape)
            else:
                source_data = LOAD_FUNCTIONS[source_fmt](source_path)

            if source_data is None:
                print(f"    跳过 {source_fmt.upper()} (加载失败)")
                continue

            # 检查数据是否为2D
            if source_data.ndim != 2:
                print(f"    跳过 {source_fmt.upper()} (数据不是2D: {source_data.shape})")
                continue

            print(f"    加载成功: {source_data.shape}")
            source_data_cache[source_fmt] = source_data

        except Exception as e:
            print(f"    加载失败: {e}")
            continue

    print(f"\n  成功加载 {len(source_data_cache)} 种格式的数据")

    # 执行转换
    print("\n  开始转换...")
    for source_fmt, source_data in source_data_cache.items():
        target_formats = [f for f in formats if f != source_fmt]
        for target_fmt in target_formats:
            # 生成文件名
            converted_name = f'{base_name}_from_{source_fmt}_to_{target_fmt}'
            output_path = os.path.join(output_dir, f'{converted_name}.{target_fmt}')

            # 保存转换后的数据
            try:
                if target_fmt == 'bin':
                    save_to_bin(source_data, output_path)
                else:
                    SAVE_FUNCTIONS[target_fmt](source_data, output_path)
                conversion_count += 1
                print(f"    ✓ {FORMAT_LABELS[source_fmt]} → {FORMAT_LABELS[target_fmt]}: {converted_name}.{target_fmt}")

                # 存储转换后的数据信息用于后续绘图
                key = f'{source_fmt}->{target_fmt}'
                all_converted_data[key] = {
                    'data': source_data.copy(),  # 复制数据避免引用问题
                    'path': output_path,
                    'source': source_fmt,
                    'target': target_fmt
                }
            except Exception as e:
                print(f"    ✗ {FORMAT_LABELS[source_fmt]} → {FORMAT_LABELS[target_fmt]} 失败: {e}")

    print(f"\n  完成 {conversion_count} 次转换，耗时: {time.time() - start_time:.2f}秒")

    # 5. 加载所有转换后的数据进行对比
    print("\n[4] 加载所有转换后的数据...")
    load_start = time.time()

    converted_data_dict = {}
    for key, info in all_converted_data.items():
        # 直接使用内存中的数据，避免重新加载
        data = info['data']
        if data is not None and data.ndim == 2:
            converted_data_dict[key] = data
            print(f"  ✓ {key}: {data.shape}")
        else:
            print(f"  ✗ {key}: 数据无效 (shape: {data.shape if data is not None else 'None'})")

    print(f"  加载耗时: {time.time() - load_start:.2f}秒")

    # 6. 在同一张图中画出所有转换后的数据
    print("\n[5] 生成对比图...")

    n_datasets = len(converted_data_dict)
    if n_datasets == 0:
        print("  没有可用的数据")
        return

    # 计算子图布局
    n_cols = min(5, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols

    # 调整图形大小
    fig_width = 6 * n_cols
    fig_height = 5 * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

    # 确保axes是1D数组
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    plot_idx = 0
    for key, data in converted_data_dict.items():
        source, target = key.split('->')
        title = f'{FORMAT_LABELS[source]} → {FORMAT_LABELS[target]}'

        # 显示2D数据
        im = axes[plot_idx].imshow(data, aspect='auto', cmap='jet', interpolation='bilinear')
        axes[plot_idx].set_title(f'{title}\n{data.shape}', fontsize=11, fontweight='bold')
        axes[plot_idx].set_xlabel('X (Trace)', fontsize=9)
        axes[plot_idx].set_ylabel('Y (Sample)', fontsize=9)
        axes[plot_idx].tick_params(labelsize=8)

        cbar = plt.colorbar(im, ax=axes[plot_idx])
        cbar.ax.tick_params(labelsize=8)
        cbar.set_label('Density', fontsize=8)

        # 显示统计信息
        info_text = f'Min: {data.min():.2f}\nMax: {data.max():.2f}'
        axes[plot_idx].text(0.02, 0.98, info_text,
                            transform=axes[plot_idx].transAxes,
                            fontsize=7,
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

        # 标记数据来源
        axes[plot_idx].text(0.98, 0.02, f'From {FORMAT_LABELS[source]}',
                            transform=axes[plot_idx].transAxes,
                            fontsize=7, horizontalalignment='right',
                            verticalalignment='bottom',
                            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

        plot_idx += 1

    # 隐藏多余的子图
    for idx in range(plot_idx, len(axes)):
        axes[idx].axis('off')

    plt.suptitle(f'Data Format Conversion Comparison\n{base_name}',
                 fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    fig_path = os.path.join(fig_dir, f'{base_name}_conversion_comparison.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ 对比图已保存: {fig_path}")
    plt.close()

    # 7. 生成转换摘要报告
    print("\n[6] 生成转换摘要...")

    # 按源格式分组统计
    conversion_summary = {}
    for key in converted_data_dict.keys():
        source, target = key.split('->')
        if source not in conversion_summary:
            conversion_summary[source] = []
        conversion_summary[source].append(target)

    print("\n转换摘要:")
    for source, targets in conversion_summary.items():
        print(f"  {FORMAT_LABELS[source]} → {', '.join([FORMAT_LABELS[t] for t in targets])}")

    # 保存摘要到文本文件
    summary_path = os.path.join(output_dir, 'conversion_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("数据格式转换摘要\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"基础名称: {base_name}\n")
        if bin_shape:
            f.write(f"BIN维度: {bin_shape}\n")
        f.write(f"总转换数: {len(converted_data_dict)}\n\n")
        f.write("转换列表:\n")
        for key in sorted(converted_data_dict.keys()):
            source, target = key.split('->')
            f.write(f"  {FORMAT_LABELS[source]} → {FORMAT_LABELS[target]}\n")

    print(f"  摘要已保存: {summary_path}")

    # 总结
    print("\n" + "=" * 80)
    print("转换完成！")
    print("=" * 80)
    print(f"转换数据目录: {output_dir}")
    print(f"对比图: {fig_path}")
    print(f"转换摘要: {summary_path}")
    print(f"总共完成 {len(converted_data_dict)} 次格式转换")
    if bin_shape:
        print(f"BIN数据维度: {bin_shape}")
    print("=" * 80)


if __name__ == "__main__":
    main()
