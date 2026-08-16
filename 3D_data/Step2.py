import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import os
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MultipleLocator, ScalarFormatter
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
from obspy.io.segy.segy import _read_segy
import obspy
import pandas as pd


def read_segy_data(file_path):
    """
    使用 ObsPy 读取 SEGY 数据，并转置
    原代码中的读取方式：先读取traces，再转置
    """
    print(f"正在读取 SEGY 文件: {os.path.basename(file_path)}")
    try:
        segy = _read_segy(file_path)
        traces_data = []
        for trace in segy.traces:
            traces_data.append(trace.data)
        data = np.array(traces_data)
        print(f"  读取后数据形状 (traces, samples): {data.shape}")
        return data
    except Exception as e:
        print(f"  _read_segy 失败: {e}")
        try:
            st = obspy.read(file_path, format='SEGY')
            data = np.array([tr.data for tr in st])
            print(f"  读取后数据形状 (traces, samples): {data.shape}")
            return data
        except Exception as e2:
            print(f"  obspy.read 也失败了: {e2}")
            return None


def yc_scale(D, N, dscale=1.0):
    """
    Scale the data up to the Nth dimension = sfscale axis=N.
    Parameters:
        D (ndarray): Input data.
        N (int): Number of dimension for scaling. Default is 2.
        dscale (float): Scale by this factor. Default is 1.0.
    Returns:
        D1 (ndarray): Output scaled data.
    """
    if D.size == 0:
        raise ValueError('Input data must be provided!')
    n1, n2, n3 = D.shape
    D1 = D.copy()
    if N == 1:
        for i3 in range(n3):
            for i2 in range(n2):
                D1[:, i2, i3] /= np.max(np.abs(D1[:, i2, i3]))
    elif N == 2:
        for i3 in range(n3):
            D1[:, :, i3] /= np.max(np.abs(D1[:, :, i3]))
    elif N == 3:
        D1 /= np.max(np.abs(D1))
    elif N == 0:
        D1 *= dscale
    else:
        raise ValueError('Invalid argument value N.')
    return D1


def reshape_to_3d(data_2d, dims_3d):
    """
    将2D数据重塑为3D数据
    """
    total_elements = data_2d.size
    expected_elements = dims_3d[0] * dims_3d[1] * dims_3d[2]

    print(f"\n数据重塑:")
    print(f"  原始数据形状: {data_2d.shape}, 元素数: {total_elements}")
    print(f"  目标3D形状: {dims_3d}, 目标元素数: {expected_elements}")

    if total_elements != expected_elements:
        print(f"  警告: 元素数量不匹配! 差异: {total_elements - expected_elements}")
        if total_elements > expected_elements:
            data_2d = data_2d[:expected_elements]
            print(f"  截断数据到 {expected_elements} 个元素")
        else:
            pad_size = expected_elements - total_elements
            data_2d = np.pad(data_2d, (0, pad_size), 'constant', constant_values=0)
            print(f"  补零到 {expected_elements} 个元素")

    data_3d = data_2d.reshape(dims_3d)
    print(f"  重塑后3D形状: {data_3d.shape}")
    return data_3d


def cseis():
    seis = np.concatenate(
        (np.concatenate((0.5 * np.ones([1, 40]), np.expand_dims(np.linspace(0.5, 1, 88), axis=1).transpose(),
                         np.expand_dims(np.linspace(1, 0, 88), axis=1).transpose(), np.zeros([1, 40])),
                        axis=1).transpose(),
         np.concatenate((0.25 * np.ones([1, 40]), np.expand_dims(np.linspace(0.25, 1, 88), axis=1).transpose(),
                         np.expand_dims(np.linspace(1, 0, 88), axis=1).transpose(), np.zeros([1, 40])),
                        axis=1).transpose(),
         np.concatenate((np.zeros([1, 40]), np.expand_dims(np.linspace(0, 1, 88), axis=1).transpose(),
                         np.expand_dims(np.linspace(1, 0, 88), axis=1).transpose(), np.zeros([1, 40])),
                        axis=1).transpose()), axis=1)
    return ListedColormap(seis)


def create_black_white_red_cmap():
    """创建自定义的黑-白-红色标"""
    colors = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.5), (1.0, 1.0, 1.0), (1.0, 0.5, 0.5), (1.0, 0.0, 0.0)]
    return mcolors.LinearSegmentedColormap.from_list('black_white_red', colors)


def round_to_nice(num):
    """将数字取整到最接近的整十/整百/整千等"""
    if num < 1:
        return 1
    magnitude = 10 ** (len(str(int(num))) - 1)
    base = num / magnitude
    if base <= 1.5:
        return magnitude
    elif base <= 3.5:
        return 2 * magnitude
    elif base <= 7.5:
        return 5 * magnitude
    else:
        return 10 * magnitude


def get_z_ticks(zmin, zmax, n_ticks=6):
    """获取z轴刻度，保留两位小数且第二位为0"""
    if zmin == zmax:
        return np.array([zmin])
    range_val = zmax - zmin
    raw_interval = range_val / (n_ticks - 1)
    if range_val < 0.1:
        interval = round(raw_interval / 0.01) * 0.01
    elif range_val < 0.5:
        interval = round(raw_interval / 0.02) * 0.02
    elif range_val < 1:
        interval = round(raw_interval / 0.05) * 0.05
    elif range_val < 2:
        interval = round(raw_interval / 0.1) * 0.1
    elif range_val < 5:
        interval = round(raw_interval / 0.2) * 0.2
    elif range_val < 10:
        interval = round(raw_interval / 0.5) * 0.5
    else:
        interval = round(raw_interval)
    if interval == 0:
        interval = 0.01
    start = np.ceil(zmin / interval) * interval
    end = np.floor(zmax / interval) * interval
    ticks = np.arange(start, end + interval, interval)
    if len(ticks) == 0:
        ticks = np.array([zmin, zmax])
    elif ticks[0] > zmin:
        ticks = np.insert(ticks, 0, zmin)
    elif ticks[-1] < zmax:
        ticks = np.append(ticks, zmax)
    ticks = np.round(ticks, 2)
    return ticks


def get_nice_ticks(vmin, vmax, n_ticks=6):
    """获取漂亮的刻度值，支持小数"""
    if vmin == vmax:
        return np.array([vmin])
    range_val = vmax - vmin
    raw_interval = range_val / (n_ticks - 1)
    if range_val < 1:
        precision = 10 ** (len(str(range_val).split('.')[1]) - 1) if '.' in str(range_val) else 1
        nice_interval = round(raw_interval * precision) / precision
    else:
        nice_interval = round_to_nice(raw_interval)
    start = np.ceil(vmin / nice_interval) * nice_interval if nice_interval != 0 else vmin
    end = np.floor(vmax / nice_interval) * nice_interval if nice_interval != 0 else vmax
    ticks = np.arange(start, end + nice_interval, nice_interval)
    if len(ticks) == 0:
        ticks = np.array([vmin, vmax])
    elif ticks[0] > vmin:
        ticks = np.insert(ticks, 0, vmin)
    elif ticks[-1] < vmax:
        ticks = np.append(ticks, vmax)
    return ticks


def plot3d_single(d3d, frames=None, z=None, x=None, y=None, dz=0.01, dx=0.01, dy=0.01,
                  nlevel=100, cmap='seismic', vmin=None, vmax=None, cbar_ticks=None,
                  ax=None, title=None, show_cbar=True, x_start=0, y_start=0, z_start=0, **kwargs):
    """
    绘制单个3D数据体
    """
    [nz, nx, ny] = d3d.shape
    if frames is None:
        frames = [int(nz / 2), int(nx / 2), int(ny / 2)]
    if z is None:
        z = np.arange(nz) * dz + z_start
    if x is None:
        x = np.arange(nx) * dx + x_start
    if y is None:
        y = np.arange(ny) * dy + y_start

    X, Y, Z = np.meshgrid(x, y, z)
    d3d_transposed = d3d.transpose([1, 2, 0])

    # 设置色标
    if cmap == 'seismic':
        cmap_obj = plt.cm.seismic
    elif cmap == 'cseis':
        cmap_obj = cseis()
    elif cmap == 'black_white_red':
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
        vmin = d3d.min()
    if vmax is None:
        vmax = d3d.max()

    kw = {'vmin': vmin, 'vmax': vmax, 'levels': np.linspace(vmin, vmax, nlevel), 'cmap': cmap_obj}
    kw.update(kwargs)
    if 'alpha' not in kw.keys():
        kw['alpha'] = 1.0

    if ax is None:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
    else:
        fig = ax.figure

    # 绘制三个方向的切片
    ax.contourf(X[:, :, -1], Y[:, :, -1], d3d_transposed[:, :, frames[0]].transpose(),
                zdir='z', offset=z_start, zorder=1, **kw)
    ax.contourf(X[0, :, :], d3d_transposed[:, frames[2], :], Z[0, :, :],
                zdir='y', offset=y_start, zorder=1, **kw)
    C = ax.contourf(d3d_transposed[frames[1], :, :], Y[:, -1, :], Z[:, -1, :],
                    zdir='x', offset=x_start + (nx - 1) * dx, zorder=1, **kw)

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

    # 设置刻度大小
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    ax.tick_params(axis='z', labelsize=8)

    # 设置刻度间隔
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

    # YZ平面 (在x_start + (nx-1)*dx位置)
    x_end = x_start + (nx - 1) * dx
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

    if show_cbar:
        cbar = plt.colorbar(C, ax=ax, orientation='vertical', fraction=0.04, pad=0.06)
        cbar.set_label('Amplitude', fontsize=9, fontweight='bold')
        cbar.ax.tick_params(labelsize=8)
        if cbar_ticks is not None:
            cbar.set_ticks(cbar_ticks)
        else:
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

    if title:
        ax.set_title(title, fontsize=11, fontweight='bold')

    return fig, ax, C


def load_3d_data(file_path, data_shape):
    """加载3D数据（支持多种格式）"""
    print(f"Loading 3D data: {os.path.basename(file_path)}")
    if not os.path.exists(file_path):
        print(f"  File not found: {file_path}")
        return None
    try:
        if file_path.endswith('.npy'):
            data = np.load(file_path)
        elif file_path.endswith('.mat'):
            mat_data = loadmat(file_path)
            data = mat_data['data'] if 'data' in mat_data else next(
                mat_data[key] for key in mat_data if not key.startswith('__'))
        elif file_path.endswith('.bin'):
            n_elements = data_shape[0] * data_shape[1] * data_shape[2]
            data = np.fromfile(file_path, dtype=np.float32, count=n_elements).reshape(data_shape)
        elif file_path.endswith('.dat') or file_path.endswith('.txt'):
            data_flat = np.loadtxt(file_path, dtype=np.float32)
            n_elements = len(data_flat)
            expected_elements = data_shape[0] * data_shape[1] * data_shape[2]
            if n_elements >= expected_elements:
                data = data_flat[:expected_elements].reshape(data_shape)
            else:
                print(f"  Not enough elements: {n_elements} < {expected_elements}")
                return None
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path, header=None)
            data_flat = df.values.flatten()
            n_elements = len(data_flat)
            expected_elements = data_shape[0] * data_shape[1] * data_shape[2]
            if n_elements >= expected_elements:
                data = data_flat[:expected_elements].reshape(data_shape)
            else:
                print(f"  Not enough elements: {n_elements} < {expected_elements}")
                return None
        else:
            print(f"  Unsupported format: {file_path}")
            return None
        print(f"  3D shape: {data.shape}")
        return data
    except Exception as e:
        print(f"  Load failed: {e}")
        return None


def main():
    data_dir = 'data'
    segy_file = os.path.join(data_dir, 'Seismic_0414.segy')
    output_dir = os.path.join(data_dir, 'figs')
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("SEGY vs 3D Data Comparison")
    print("=" * 70)

    # 1. 读取SEGY数据（使用Step1的方式）
    data_2d = read_segy_data(segy_file)
    if data_2d is None:
        print("SEGY data load failed, exit")
        return

    print(f"\n数据统计:")
    print(f"  形状: {data_2d.shape}")
    print(f"  总元素数: {data_2d.size}")
    print(f"  数据类型: {data_2d.dtype}")
    print(f"  数值范围: [{data_2d.min():.4f}, {data_2d.max():.4f}]")

    # 2. 获取维度信息
    n_traces, n_samples = data_2d.shape

    print("\n" + "=" * 70)
    print("Specify 3D data dimensions (for SEGY and all formats):")
    print("=" * 70)
    print(f"SEGY data shape (traces, samples): {data_2d.shape}")
    print(f"  Traces count: {n_traces}")
    print(f"  Samples count: {n_samples}")

    # 用户输入三个维度（SEGY和其他格式共用）
    print("\nPlease enter the three dimensions for 3D data:")
    dim_z = int(input("Z dimension (Time, default from samples): ") or n_samples)
    dim_x = int(input("X dimension (Inline): "))
    dim_y = int(input("Y dimension (Crossline): "))

    data_shape_3d = (dim_z, dim_x, dim_y)
    print(f"\n3D data shape (Time, Inline, Crossline): {data_shape_3d}")

    # 3. 重塑SEGY数据为3D (完全按照Step1的方式)
    total_elements = data_2d.size
    expected_elements = dim_z * dim_x * dim_y

    print(f"\n总元素数: {total_elements}")
    print(f"期望元素数: {expected_elements}")

    if total_elements != expected_elements:
        print(f"  警告: 元素数量不匹配! 差异: {total_elements - expected_elements}")
        confirm = input("是否继续? (y/n, 默认n): ").strip().lower()
        if confirm != 'y':
            print("程序退出")
            return

    # 按照Step1的方式重塑: (Y, X, Z) -> transpose -> (Z, X, Y)
    dims_3d = (dim_y, dim_x, dim_z)  # (Y, X, Z) 对应 (Crossline, Inline, Time)
    print(f"\n重塑维度 (Y, X, Z): {dims_3d}")

    data_3d_temp = reshape_to_3d(data_2d, dims_3d)
    segy_3d = np.transpose(data_3d_temp, (2, 1, 0))  # (Z, X, Y) = (Time, Inline, Crossline)
    print(f"转置后SEGY 3D形状: {segy_3d.shape}")

    # 4. 加载其他格式的3D数据（使用相同的维度）
    data_3d_dict = {'SEGY': segy_3d}
    formats = {
        'NPY': 'Seismic_0414_3D_original.npy',
        'MAT': 'Seismic_0414_3D_original.mat',
        'BIN': 'Seismic_0414_3D_original.bin',
        'DAT': 'Seismic_0414_3D_original.dat',
        'TXT': 'Seismic_0414_3D_original.txt',
        'CSV': 'Seismic_0414_3D_original.csv'
    }

    print("\n" + "=" * 70)
    print("Loading 3D data files:")
    print("=" * 70)
    for fmt, filename in formats.items():
        file_path = os.path.join(data_dir, filename)
        data = load_3d_data(file_path, data_shape_3d)
        if data is not None:
            data_3d_dict[fmt] = data

    # 5. 设置绘图参数
    print("\n" + "=" * 70)
    print("Plot parameters:")
    print("=" * 70)

    # 归一化选项
    print("\nNormalization options:")
    print("  0: No normalization (default)")
    print("  1: Normalize along 1st dimension")
    print("  2: Normalize along 2nd dimension")
    print("  3: Global normalization")
    norm_choice = input("Enter normalization option (default 0): ").strip()
    norm_choice = int(norm_choice) if norm_choice else 0

    if norm_choice != 0:
        print(f"  Applying normalization (N={norm_choice})")
        # 对所有3D数据应用归一化
        for fmt in data_3d_dict.keys():
            data_3d_dict[fmt] = yc_scale(data_3d_dict[fmt], norm_choice)
        print("  Normalization applied to all data")

    # 坐标轴起始点
    print("\nAxis start values (optional, press Enter to use 0):")
    x_start_input = input("Enter X axis (Inline) start value (default 0): ").strip()
    x_start = float(x_start_input) if x_start_input else 0.0

    y_start_input = input("Enter Y axis (Crossline) start value (default 0): ").strip()
    y_start = float(y_start_input) if y_start_input else 0.0

    z_start_input = input("Enter Z axis (Time) start value (default 0): ").strip()
    z_start = float(z_start_input) if z_start_input else 0.0

    print(f"  X start: {x_start}, Y start: {y_start}, Z start: {z_start}")

    dz_input = input("\nEnter z-axis time sampling interval (default 0.004): ").strip()
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

    # 6. 选择切片位置
    print("\n" + "=" * 70)
    print("Select slice positions:")
    print("=" * 70)
    print("Z slice (Time), X slice (Inline), Y slice (Crossline)")
    try:
        slice_z = int(input(f"Enter Z slice position (0-{dim_z - 1}, default middle): ") or dim_z // 2)
        slice_x = int(input(f"Enter X slice position (0-{dim_x - 1}, default middle): ") or dim_x // 2)
        slice_y = int(input(f"Enter Y slice position (0-{dim_y - 1}, default middle): ") or dim_y // 2)
    except:
        slice_z, slice_x, slice_y = dim_z // 2, dim_x // 2, dim_y // 2

    frames = [slice_z, slice_x, slice_y]
    print(f"\nSlice positions: Z={slice_z}, X={slice_x}, Y={slice_y}")

    # 7. 绘制对比图 - 两行四列布局
    n_formats = len(data_3d_dict)
    n_cols = 4
    n_rows = 2

    fig = plt.figure(figsize=(28, 12))

    # 获取所有格式的keys
    format_keys = list(data_3d_dict.keys())

    for idx, fmt in enumerate(format_keys):
        if idx >= n_rows * n_cols:
            break

        data_3d = data_3d_dict[fmt]
        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection='3d')

        # 使用plot3d_single绘制，每个子图都显示色棒
        _, _, _ = plot3d_single(
            data_3d,
            frames=frames,
            dz=dz,
            dx=1,
            dy=1,
            nlevel=80,
            cmap=cmap_choice,
            vmin=vmin,
            vmax=vmax,
            cbar_ticks=cbar_ticks,
            ax=ax,
            title=f'{fmt} format\nShape: {data_3d.shape}',
            show_cbar=True,
            x_start=x_start,
            y_start=y_start,
            z_start=z_start
        )

    # 如果子图少于8个，隐藏多余的子图
    for idx in range(len(format_keys), n_rows * n_cols):
        fig.add_subplot(n_rows, n_cols, idx + 1).set_visible(False)

    plt.tight_layout()
    output_file = os.path.join(output_dir, '3d_formats_comparison.png')
    plt.savefig(output_file, bbox_inches='tight', pad_inches=0.2, dpi=300)
    print(f"\nComparison figure saved: {output_file}")
    plt.close()

    print("\n" + "=" * 70)
    print("Processing complete!")
    print("=" * 70)
    print(f"SEGY 2D shape: {data_2d.shape}")
    print(f"SEGY 3D shape: {segy_3d.shape}")
    print(f"3D data shape: {data_shape_3d}")
    print(f"Loaded formats: {list(data_3d_dict.keys())}")
    print(f"Slice positions: Z={slice_z}, X={slice_x}, Y={slice_y}")
    print(f"Colormap: {cmap_choice}")
    print(f"Normalization: {norm_choice if norm_choice != 0 else 'None'}")
    print(f"Axis starts: X={x_start}, Y={y_start}, Z={z_start}")
    print(f"Output: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()