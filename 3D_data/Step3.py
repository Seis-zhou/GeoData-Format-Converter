import numpy as np
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt
import os
import pandas as pd
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MultipleLocator, ScalarFormatter


def load_from_npy(file_path):
    """从NPY加载数据"""
    try:
        return np.load(file_path)
    except Exception as e:
        print(f"    加载NPY失败: {e}")
        return None


def load_from_bin(file_path, shape=None):
    """从BIN加载数据 - 需要手动指定形状"""
    try:
        data = np.fromfile(file_path, dtype=np.float32)
        if shape is not None:
            try:
                return data.reshape(shape)
            except Exception as e:
                print(f"    错误: 无法重塑为 {shape}，数据长度 {len(data)} 不匹配")
                total_elements = shape[0] * shape[1] * shape[2]
                if len(data) != total_elements:
                    print(f"    数据长度 {len(data)} != 期望 {total_elements}")
                return None
        else:
            print(f"    警告: 未指定形状，数据长度: {len(data)}")
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
        try:
            data = np.loadtxt(file_path, dtype=np.float32)
            return data
        except:
            data_list = []
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        values = line.strip().split()
                        if values:
                            data_list.extend([float(v) for v in values])
                    except:
                        continue
            if data_list:
                return np.array(data_list, dtype=np.float32)
            else:
                print(f"    无法解析DAT文件")
                return None
    except Exception as e:
        print(f"    加载DAT失败: {e}")
        return None


def load_from_txt(file_path):
    """从TXT加载数据"""
    try:
        try:
            data = np.loadtxt(file_path, dtype=np.float32)
            return data
        except:
            data_list = []
            with open(file_path, 'r') as f:
                for line in f:
                    try:
                        values = line.strip().split()
                        if values:
                            data_list.extend([float(v) for v in values])
                    except:
                        continue
            if data_list:
                return np.array(data_list, dtype=np.float32)
            else:
                print(f"    无法解析TXT文件")
                return None
    except Exception as e:
        print(f"    加载TXT失败: {e}")
        return None


def load_from_csv(file_path):
    """从CSV加载数据"""
    try:
        try:
            df = pd.read_csv(file_path, header=None, dtype=np.float32)
            data = df.values.flatten()
            return data.astype(np.float32)
        except Exception as e1:
            print(f"    第一种方式失败: {e1}")
            try:
                df = pd.read_csv(file_path, header=None, low_memory=False)
                for col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df.dropna()
                data = df.values.flatten()
                return data.astype(np.float32)
            except Exception as e2:
                print(f"    第二种方式失败: {e2}")
                data_list = []
                with open(file_path, 'r') as f:
                    for line in f:
                        try:
                            values = line.strip().split(',')
                            if values:
                                data_list.extend([float(v) for v in values if v])
                        except:
                            continue
                if data_list:
                    return np.array(data_list, dtype=np.float32)
                else:
                    print(f"    无法解析CSV文件")
                    return None
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
    np.savetxt(file_path, data.flatten(), fmt='%.6f', delimiter='\t')


def save_to_txt(data, file_path):
    """保存为TXT格式"""
    np.savetxt(file_path, data.flatten(), fmt='%.6f', delimiter='\t')


def save_to_csv(data, file_path):
    """保存为CSV格式"""
    flat_data = data.flatten()
    df = pd.DataFrame({'data': flat_data})
    df.to_csv(file_path, index=False, header=False)


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

FORMAT_LABELS = {
    'npy': 'NPY',
    'bin': 'BIN',
    'mat': 'MAT',
    'dat': 'DAT',
    'txt': 'TXT',
    'csv': 'CSV'
}


def get_user_input_3d_shape():
    """获取用户输入的3D维度"""
    while True:
        try:
            user_input = input("请输入3D数据的维度 (例如: 426,601,271 即 Time, Inline, Crossline): ")
            if user_input.lower() == 'skip':
                return None
            parts = user_input.split(',')
            if len(parts) == 3:
                dim1 = int(parts[0].strip())
                dim2 = int(parts[1].strip())
                dim3 = int(parts[2].strip())
                if dim1 > 0 and dim2 > 0 and dim3 > 0:
                    return (dim1, dim2, dim3)
                else:
                    print("  维度必须为正整数")
            else:
                print("  请输入三个正整数，用逗号分隔")
        except ValueError:
            print("  输入格式错误，请输入三个正整数，用逗号分隔")
        except KeyboardInterrupt:
            print("\n  用户取消输入")
            return None


def plot_3d_comparison(converted_data_dict, output_dir, base_name,
                       dz=0.004, cmap='seismic', vmin=None, vmax=None,
                       cbar_ticks=None, x_start=0, y_start=0, z_start=0,
                       norm_choice=0):
    """绘制3D数据转换对比图"""
    n_datasets = len(converted_data_dict)
    if n_datasets == 0:
        print("  没有可用的数据")
        return None

    # 动态计算行列数
    n_cols = min(5, n_datasets)
    n_rows = (n_datasets + n_cols - 1) // n_cols
    max_plots = n_rows * n_cols

    fig_width = 7 * n_cols
    fig_height = 6 * n_rows
    fig = plt.figure(figsize=(fig_width, fig_height))

    plot_idx = 0
    for key, info in converted_data_dict.items():
        if plot_idx >= max_plots:
            break

        data = info['data']
        source, target = key.split('->')
        title = f'{FORMAT_LABELS[source]} → {FORMAT_LABELS[target]}'

        if data.ndim != 3:
            print(f"  跳过 {key}: 数据不是3D (shape: {data.shape})")
            continue

        ax = fig.add_subplot(n_rows, n_cols, plot_idx + 1, projection='3d')

        # 调用完整的plot3d_single函数
        from Step2 import plot3d_single  # 或者直接复制plot3d_single函数到这里

        # 如果不想import，可以在这里直接使用下面的完整绘图代码
        # ========== 完整的绘图代码（与plot3d_single一致） ==========
        [nz, nx, ny] = data.shape
        frames = [nz // 2, nx // 2, ny // 2]

        z = np.arange(nz) * dz + z_start
        x = np.arange(nx) + x_start
        y = np.arange(ny) + y_start

        X, Y, Z = np.meshgrid(x, y, z)
        d3d_transposed = data.transpose([1, 2, 0])

        # 设置色标
        if cmap == 'seismic':
            cmap_obj = plt.cm.seismic
        elif cmap == 'cseis':
            from Step2 import cseis
            cmap_obj = cseis()
        elif cmap == 'black_white_red':
            from Step2 import create_black_white_red_cmap
            cmap_obj = create_black_white_red_cmap()
        elif cmap == 'gray':
            cmap_obj = plt.cm.gray
        elif cmap == 'jet':
            cmap_obj = plt.cm.jet
        elif cmap == 'RdBu':
            cmap_obj = plt.cm.RdBu
        elif cmap == 'coolwarm':
            cmap_obj = plt.cm.coolwarm
        else:
            cmap_obj = plt.cm.seismic

        if vmin is None:
            vmin = data.min()
        if vmax is None:
            vmax = data.max()

        kw = {'vmin': vmin, 'vmax': vmax, 'levels': 80, 'cmap': cmap_obj, 'alpha': 0.8}

        # 绘制三个方向的切片
        ax.contourf(X[:, :, -1], Y[:, :, -1], d3d_transposed[:, :, frames[0]].transpose(),
                    zdir='z', offset=z_start, zorder=1, **kw)
        ax.contourf(X[0, :, :], d3d_transposed[:, frames[2], :], Z[0, :, :],
                    zdir='y', offset=y_start, zorder=1, **kw)
        C = ax.contourf(d3d_transposed[frames[1], :, :], Y[:, -1, :], Z[:, -1, :],
                        zdir='x', offset=x_start + (nx - 1), zorder=1, **kw)

        # 设置坐标轴标签
        ax.set_xlabel("Inline", fontsize=10, fontweight='bold')
        ax.set_ylabel("Crossline", fontsize=10, fontweight='bold')
        if dz != 1:
            ax.set_zlabel("Time (s)", fontsize=10, fontweight='bold')
        else:
            ax.set_zlabel("Time Sample", fontsize=10, fontweight='bold')

        # 设置坐标轴范围
        xmin, xmax = X.min(), X.max()
        ymin, ymax = Y.min(), Y.max()
        zmin, zmax = Z.min(), Z.max()
        ax.set(xlim=[xmin, xmax], ylim=[ymin, ymax], zlim=[zmin, zmax])
        ax.invert_zaxis()

        # 设置刻度
        ax.tick_params(axis='x', labelsize=8)
        ax.tick_params(axis='y', labelsize=8)
        ax.tick_params(axis='z', labelsize=8)

        # 设置刻度间隔
        from Step2 import round_to_nice, get_z_ticks
        x_interval = round_to_nice(nx // 6)
        y_interval = round_to_nice(ny // 6)
        ax.xaxis.set_major_locator(MultipleLocator(x_interval))
        ax.yaxis.set_major_locator(MultipleLocator(y_interval))

        # z轴刻度
        if dz != 1:
            z_ticks = get_z_ticks(zmin, zmax, 6)
            ax.set_zticks(z_ticks)
            ax.zaxis.set_major_formatter(ScalarFormatter(useOffset=False))
            ax.zaxis.get_major_formatter().set_scientific(False)
        else:
            z_interval = round_to_nice(nz // 6)
            ax.zaxis.set_major_locator(MultipleLocator(z_interval))

        # ==================== 添加蓝色十字基准线 ====================
        slice_z_idx = frames[0]
        slice_x_idx = frames[1]
        slice_y_idx = frames[2]

        x_slice = x[slice_x_idx]
        y_slice = y[slice_y_idx]
        z_slice = z[slice_z_idx]

        zorder_line = 10

        # XY平面 (在z_start位置)
        ax.plot([x_slice, x_slice], [y[0], y[-1]], [z_start, z_start], 'b-',
                linewidth=2, alpha=0.9, zorder=zorder_line)
        ax.plot([x[0], x[-1]], [y_slice, y_slice], [z_start, z_start], 'b-',
                linewidth=2, alpha=0.9, zorder=zorder_line)
        ax.plot([x[0], x[-1]], [y[0], y[0]], [z_start, z_start], 'b-',
                linewidth=1, alpha=0.5, zorder=zorder_line)
        ax.plot([x[0], x[-1]], [y[-1], y[-1]], [z_start, z_start], 'b-',
                linewidth=1, alpha=0.5, zorder=zorder_line)
        ax.plot([x[0], x[0]], [y[0], y[-1]], [z_start, z_start], 'b-',
                linewidth=1, alpha=0.5, zorder=zorder_line)
        ax.plot([x[-1], x[-1]], [y[0], y[-1]], [z_start, z_start], 'b-',
                linewidth=1, alpha=0.5, zorder=zorder_line)

        # XZ平面 (在y_start位置)
        ax.plot([x_slice, x_slice], [y_start, y_start], [z[0], z[-1]], 'b-',
                linewidth=2, alpha=0.9, zorder=zorder_line)
        ax.plot([x[0], x[-1]], [y_start, y_start], [z_slice, z_slice], 'b-',
                linewidth=2, alpha=0.9, zorder=zorder_line)
        ax.plot([x[0], x[-1]], [y_start, y_start], [z[0], z[0]], 'b-',
                linewidth=1, alpha=0.5, zorder=zorder_line)
        ax.plot([x[0], x[-1]], [y_start, y_start], [z[-1], z[-1]], 'b-',
                linewidth=1, alpha=0.5, zorder=zorder_line)
        ax.plot([x[0], x[0]], [y_start, y_start], [z[0], z[-1]], 'b-',
                linewidth=1, alpha=0.5, zorder=zorder_line)
        ax.plot([x[-1], x[-1]], [y_start, y_start], [z[0], z[-1]], 'b-',
                linewidth=1, alpha=0.5, zorder=zorder_line)

        # YZ平面 (在x_start + (nx-1)位置)
        x_end = x_start + (nx - 1)
        ax.plot([x_end, x_end], [y_slice, y_slice], [z[0], z[-1]], 'b-',
                linewidth=2, alpha=0.9, zorder=zorder_line)
        ax.plot([x_end, x_end], [y[0], y[-1]], [z_slice, z_slice], 'b-',
                linewidth=2, alpha=0.9, zorder=zorder_line)
        ax.plot([x_end, x_end], [y[0], y[0]], [z[0], z[0]], 'b-',
                linewidth=1, alpha=0.5, zorder=zorder_line)
        ax.plot([x_end, x_end], [y[0], y[0]], [z[-1], z[-1]], 'b-',
                linewidth=1, alpha=0.5, zorder=zorder_line)
        ax.plot([x_end, x_end], [y[-1], y[-1]], [z[0], z[0]], 'b-',
                linewidth=1, alpha=0.5, zorder=zorder_line)
        ax.plot([x_end, x_end], [y[-1], y[-1]], [z[-1], z[-1]], 'b-',
                linewidth=1, alpha=0.5, zorder=zorder_line)

        ax.view_init(elev=20, azim=-65)

        # Colorbar
        cbar = plt.colorbar(C, ax=ax, orientation='vertical', fraction=0.04, pad=0.06)
        cbar.set_label('Amplitude', fontsize=9, fontweight='bold')
        cbar.ax.tick_params(labelsize=8)

        # 设置色棒刻度
        if cbar_ticks is not None:
            cbar.set_ticks(cbar_ticks)
        else:
            from Step2 import get_nice_ticks
            current_ticks = cbar.get_ticks()
            integer_ticks = [t for t in current_ticks if abs(t - round(t)) < 1e-6 and t >= vmin and t <= vmax]
            if len(integer_ticks) < 3:
                nice_ticks = get_nice_ticks(vmin, vmax, 5)
                integer_ticks = [t for t in nice_ticks if abs(t - round(t)) < 1e-6]
                if len(integer_ticks) < 3:
                    int_vmin = np.floor(vmin) if vmin < 0 else np.ceil(vmin)
                    int_vmax = np.floor(vmax) if vmax < 0 else np.ceil(vmax)
                    step = 1
                    if abs(int_vmax - int_vmin) > 20:
                        step = round_to_nice(abs(int_vmax - int_vmin) / 5)
                    integer_ticks = np.arange(int_vmin, int_vmax + step, step)
            integer_ticks = [t for t in integer_ticks if t >= vmin and t <= vmax]
            if len(integer_ticks) >= 2:
                cbar.set_ticks(integer_ticks)

        # 标记数据来源
        ax.text2D(0.98, 0.02, f'From {FORMAT_LABELS[source]}',
                  transform=ax.transAxes, fontsize=7,
                  horizontalalignment='right', verticalalignment='bottom',
                  bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))

        ax.set_title(f'{title}\nShape: {data.shape}', fontsize=11, fontweight='bold')
        plot_idx += 1

    # 隐藏多余的子图
    for idx in range(plot_idx, max_plots):
        fig.add_subplot(n_rows, n_cols, idx + 1).set_visible(False)

    plt.tight_layout()
    fig_path = os.path.join(output_dir, f'{base_name}_3d_conversion_comparison.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ 3D对比图已保存: {fig_path}")
    plt.close()

    return fig_path


def main():
    # 配置 - 所有数据保存在data文件夹
    data_dir = 'data'
    output_dir = data_dir  # 直接使用data文件夹
    fig_dir = os.path.join(data_dir, 'figs')  # 图片保存在data/figs
    os.makedirs(fig_dir, exist_ok=True)

    base_name = 'Seismic_0414_3D_original'

    print("=" * 80)
    print("3D数据格式互转工具")
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

    # 3. 设置3D数据维度
    print("\n[2] 设置3D数据维度...")
    print("  提示: 需要指定 (Time, Inline, Crossline) 三个维度")
    print("  例如: 426,601,271")

    data_shape = get_user_input_3d_shape()
    if data_shape is None:
        print("  错误: 未指定数据维度，程序退出")
        return

    print(f"  使用维度: {data_shape} (Time, Inline, Crossline)")

    # 4. 从每种格式加载数据，然后转换成其他格式
    print("\n[3] 格式转换...")
    start_time = time.time()

    all_converted_data = {}
    conversion_count = 0

    # 先加载所有源数据到内存
    source_data_cache = {}

    for source_fmt, source_path in source_files.items():
        print(f"\n  从 {FORMAT_LABELS[source_fmt]} 加载数据...")

        try:
            if source_fmt == 'bin':
                source_data = load_from_bin(source_path, data_shape)
            else:
                source_data = LOAD_FUNCTIONS[source_fmt](source_path)

                if source_data is not None and source_data.ndim == 1:
                    try:
                        total_elements = data_shape[0] * data_shape[1] * data_shape[2]
                        if len(source_data) == total_elements:
                            source_data = source_data.reshape(data_shape)
                            print(f"    重塑为3D: {source_data.shape}")
                        elif len(source_data) > total_elements:
                            source_data = source_data[:total_elements].reshape(data_shape)
                            print(f"    截断并重塑为3D: {source_data.shape}")
                        else:
                            padded = np.pad(source_data, (0, total_elements - len(source_data)), 'constant')
                            source_data = padded.reshape(data_shape)
                            print(f"    补零并重塑为3D: {source_data.shape}")
                    except Exception as e:
                        print(f"    重塑失败: {e}")
                        continue

            if source_data is None:
                print(f"    跳过 {source_fmt.upper()} (加载失败)")
                continue

            if source_data.ndim != 3:
                print(f"    跳过 {source_fmt.upper()} (数据不是3D: {source_data.shape})")
                continue

            if source_data.shape != data_shape:
                print(f"    形状不匹配: {source_data.shape} != {data_shape}")
                try:
                    total_elements = data_shape[0] * data_shape[1] * data_shape[2]
                    if source_data.size == total_elements:
                        source_data = source_data.reshape(data_shape)
                        print(f"    重塑成功: {source_data.shape}")
                    else:
                        print(f"    元素数量不匹配: {source_data.size} != {total_elements}")
                        continue
                except Exception as e:
                    print(f"    重塑失败: {e}")
                    continue

            print(f"    加载成功: {source_data.shape}")
            print(f"    数值范围: [{source_data.min():.4f}, {source_data.max():.4f}]")
            source_data_cache[source_fmt] = source_data

        except Exception as e:
            print(f"    加载失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n  成功加载 {len(source_data_cache)} 种格式的数据")

    # # 执行转换
    # print("\n  开始转换...")
    # for source_fmt, source_data in source_data_cache.items():
    #     target_formats = [f for f in formats if f != source_fmt]
    #     for target_fmt in target_formats:
    #         # 生成文件名 - 保存在data文件夹
    #         converted_name = f'{base_name}_from_{source_fmt}_to_{target_fmt}'
    #         output_path = os.path.join(output_dir, f'{converted_name}.{target_fmt}')
    #
    #         try:
    #             if target_fmt == 'bin':
    #                 save_to_bin(source_data, output_path)
    #             else:
    #                 SAVE_FUNCTIONS[target_fmt](source_data, output_path)
    #             conversion_count += 1
    #             print(f"    ✓ {FORMAT_LABELS[source_fmt]} → {FORMAT_LABELS[target_fmt]}: {converted_name}.{target_fmt}")
    #
    #             key = f'{source_fmt}->{target_fmt}'
    #             all_converted_data[key] = {
    #                 'data': source_data.copy(),
    #                 'path': output_path,
    #                 'source': source_fmt,
    #                 'target': target_fmt
    #             }
    #         except Exception as e:
    #             print(f"    ✗ {FORMAT_LABELS[source_fmt]} → {FORMAT_LABELS[target_fmt]} 失败: {e}")
    #
    # print(f"\n  完成 {conversion_count} 次转换，耗时: {time.time() - start_time:.2f}秒")

    # 5. 加载所有转换后的数据（从文件夹重新加载）
    print("\n[4] 加载所有转换后的数据...")
    load_start = time.time()

    converted_data_dict = {}

    # 扫描data文件夹，查找所有转换后的文件
    for fmt in formats:
        for source_fmt in formats:
            if source_fmt == fmt:
                continue
            # 构建文件名
            converted_name = f'{base_name}_from_{source_fmt}_to_{fmt}'
            file_path = os.path.join(data_dir, f'{converted_name}.{fmt}')

            if os.path.exists(file_path):
                print(f"  加载: {converted_name}.{fmt}")
                try:
                    if fmt == 'bin':
                        data = load_from_bin(file_path, data_shape)
                    else:
                        data = LOAD_FUNCTIONS[fmt](file_path)

                    if data is not None:
                        if data.ndim == 1:
                            try:
                                data = data.reshape(data_shape)
                            except:
                                print(f"    警告: 无法重塑 {converted_name}.{fmt}")
                                continue

                        if data.ndim == 3 and data.shape == data_shape:
                            key = f'{source_fmt}->{fmt}'
                            converted_data_dict[key] = {
                                'data': data,
                                'path': file_path,
                                'source': source_fmt,
                                'target': fmt
                            }
                            print(f"    ✓ {key}: {data.shape}")
                        else:
                            print(f"    ✗ {converted_name}.{fmt}: 形状不匹配 {data.shape} != {data_shape}")
                except Exception as e:
                    print(f"    ✗ 加载失败 {converted_name}.{fmt}: {e}")

    print(f"  加载耗时: {time.time() - load_start:.2f}秒")
    print(f"  找到 {len(converted_data_dict)} 个转换后的数据")

    # 在main函数中，调用plot_3d_comparison之前添加参数设置
    print("\n[4.5] 设置绘图参数...")
    dz_input = input("Enter z-axis time sampling interval (default 0.004): ").strip()
    dz = float(dz_input) if dz_input else 0.004

    print("\nColormap options: seismic(default), cseis, black_white_red, gray, jet, RdBu, coolwarm")
    cmap_choice = input("Enter colormap name (default seismic): ").strip()
    cmap_choice = cmap_choice if cmap_choice else 'seismic'

    print("\nColorbar range (optional, press Enter to skip):")
    vmin_input = input("Enter colorbar min value (default data min): ").strip()
    vmin = float(vmin_input) if vmin_input else None
    vmax_input = input("Enter colorbar max value (default data max): ").strip()
    vmax = float(vmax_input) if vmax_input else None

    print("\nColorbar ticks (optional, comma separated, e.g., -1,0,1):")
    ticks_input = input("Enter colorbar ticks: ").strip()
    cbar_ticks = [float(t.strip()) for t in ticks_input.split(',')] if ticks_input else None

    print("\nAxis start values (optional, press Enter to use 0):")
    x_start_input = input("Enter X axis (Inline) start value (default 0): ").strip()
    x_start = float(x_start_input) if x_start_input else 0.0
    y_start_input = input("Enter Y axis (Crossline) start value (default 0): ").strip()
    y_start = float(y_start_input) if y_start_input else 0.0
    z_start_input = input("Enter Z axis (Time) start value (default 0): ").strip()
    z_start = float(z_start_input) if z_start_input else 0.0

    # 6. 生成3D对比图
    print("\n[5] 生成3D对比图...")
    fig_path = None
    if converted_data_dict:
        fig_path = plot_3d_comparison(
            converted_data_dict, fig_dir, base_name,
            dz=dz, cmap=cmap_choice, vmin=vmin, vmax=vmax,
            cbar_ticks=cbar_ticks, x_start=x_start, y_start=y_start, z_start=z_start
        )
        if fig_path:
            print(f"  对比图已保存: {fig_path}")
    else:
        print("  没有可用的数据生成对比图")

    # 7. 输出汇总信息
    print("\n" + "=" * 80)
    print("转换完成！")
    print("=" * 80)
    print(f"转换数据目录: {output_dir}")
    print(f"对比图目录: {fig_dir}")
    print(f"总共完成 {len(converted_data_dict)} 次格式转换")
    print(f"数据维度: {data_shape} (Time, Inline, Crossline)")

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

    print("=" * 80)


if __name__ == "__main__":
    main()