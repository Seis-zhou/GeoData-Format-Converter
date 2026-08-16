import numpy as np
import os
from scipy.io import savemat
import pandas as pd
from obspy.io.segy.segy import _read_segy
import obspy


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


def sample_with_interval(data_3d, interval=5):
    """
    沿三个维度以固定间隔采样
    """
    print(f"\n数据采样 (间隔={interval}):")
    print(f"  原始3D形状: {data_3d.shape}")

    sampled_data = data_3d[::interval, ::interval, ::interval]

    print(f"  实际采样后形状: {sampled_data.shape}")
    print(f"  原始数据点: {data_3d.size}")
    print(f"  采样后数据点: {sampled_data.size}")
    print(f"  压缩比例: {sampled_data.size / data_3d.size * 100:.2f}%")

    return sampled_data


def save_to_npy(data, filepath):
    """保存为NPY格式"""
    np.save(filepath, data)
    print(f"  ✓ NPY: {os.path.basename(filepath)}")


def save_to_bin(data, filepath):
    """保存为二进制格式"""
    data.astype(np.float32).tofile(filepath)
    print(f"  ✓ BIN: {os.path.basename(filepath)}")


def save_to_mat(data, filepath):
    """保存为MATLAB格式"""
    savemat(filepath, {'data': data})
    print(f"  ✓ MAT: {os.path.basename(filepath)}")


def save_to_dat(data, filepath):
    """保存为DAT文本格式"""
    np.savetxt(filepath, data.flatten(), fmt='%.6f', delimiter='\t')
    print(f"  ✓ DAT: {os.path.basename(filepath)}")


def save_to_txt(data, filepath):
    """保存为TXT格式"""
    np.savetxt(filepath, data.flatten(), fmt='%.6f', delimiter='\t')
    print(f"  ✓ TXT: {os.path.basename(filepath)}")


def save_to_csv(data, filepath):
    """保存为CSV格式"""
    flat_data = data.flatten()
    df = pd.DataFrame({'data': flat_data})
    df.to_csv(filepath, index=False, header=False, encoding='utf-8-sig')
    print(f"  ✓ CSV: {os.path.basename(filepath)}")


def save_all_formats(data, output_dir, base_name):
    """
    保存所有格式的数据
    """
    # 直接使用output_dir，不创建子目录
    print(f"\n保存数据到: {output_dir}")
    print(f"  数据形状: {data.shape}")
    print(f"  数据点数: {data.size}")

    # 1. NPY格式
    save_to_npy(data, os.path.join(output_dir, f'{base_name}.npy'))

    # 2. BIN格式
    save_to_bin(data, os.path.join(output_dir, f'{base_name}.bin'))

    # 3. MAT格式
    save_to_mat(data, os.path.join(output_dir, f'{base_name}.mat'))

    # 4. DAT格式
    save_to_dat(data, os.path.join(output_dir, f'{base_name}.dat'))

    # 5. TXT格式
    save_to_txt(data, os.path.join(output_dir, f'{base_name}.txt'))

    # 6. CSV格式
    save_to_csv(data, os.path.join(output_dir, f'{base_name}.csv'))


def main():
    # 设置文件路径
    data_dir = 'data'
    segy_file = os.path.join(data_dir, 'Seismic_0414.segy')

    # 输出目录直接使用SEGY文件所在目录
    output_dir = data_dir

    print("=" * 70)
    print("3D SEGY数据处理程序")
    print("=" * 70)

    # 1. 检查文件是否存在
    if not os.path.exists(segy_file):
        print(f"错误: 文件不存在 - {segy_file}")
        return

    # 2. 读取SEGY数据（带转置）
    data_2d = read_segy_data(segy_file)
    if data_2d is None:
        print("读取失败，程序退出")
        return

    # 获取维度信息
    n_traces, n_samples = data_2d.shape

    print(f"\n数据统计:")
    print(f"  形状: {data_2d.shape}")
    print(f"  总元素数: {data_2d.size}")
    print(f"  数据类型: {data_2d.dtype}")
    print(f"  数值范围: [{data_2d.min():.4f}, {data_2d.max():.4f}]")
    print(f"  Traces数量: {n_traces}")
    print(f"  Samples数量: {n_samples}")

    # 3. 询问是否采样
    print("\n" + "=" * 70)
    sample_choice = input("是否对数据进行采样? (y/n, 默认n): ").strip().lower()

    if sample_choice == 'y':
        interval_input = input("请输入采样间隔 (默认5): ").strip()
        interval = int(interval_input) if interval_input else 5
        print(f"  采样间隔: {interval}")
    else:
        interval = None
        print("  不进行采样")

    # 4. 指定3D维度 (Z维度自动使用samples数量)
    print("\n" + "=" * 70)
    print("请指定3D数据的维度大小:")
    print("=" * 70)
    print(f"Z维度 (Time Samples) 自动设置为: {n_samples}")

    total_elements = data_2d.size
    print(f"总元素数: {total_elements}")
    print(f"Traces数量: {n_traces}")

    # Z维度固定为samples数量
    dim_z = n_samples

    # 用户输入X和Y维度
    dim_x = int(input("请输入X维度大小 (Inline, 第二维): "))
    dim_y = int(input("请输入Y维度大小 (Crossline, 第三维): "))

    # 验证维度乘积是否匹配
    expected_elements = dim_z * dim_x * dim_y
    print(f"\n维度验证:")
    print(f"  Z(Time) × X(Inline) × Y(Crossline) = {dim_z} × {dim_x} × {dim_y} = {expected_elements}")
    print(f"  总元素数: {total_elements}")
    print(f"  差异: {total_elements - expected_elements}")

    if total_elements != expected_elements:
        print(f"  警告: 元素数量不匹配!")
        print(f"  建议: {n_traces} 个traces应该对应 X × Y = {dim_x} × {dim_y} = {dim_x * dim_y}")
        print(f"  当前 X × Y = {dim_x * dim_y}, 需要 = {n_traces}")
        confirm = input("是否继续? (y/n, 默认n): ").strip().lower()
        if confirm != 'y':
            print("程序退出")
            return

    dims_3d = (dim_y, dim_x, dim_z)  # (Y, X, Z) 对应 (Crossline, Inline, Time)

    # 5. 重塑为3D数据
    data_3d = reshape_to_3d(data_2d, dims_3d)
    data_3d = np.transpose(data_3d, (2, 1, 0))  # (Z, X, Y) = (Time, Inline, Crossline)
    print(f"转换后的立方体维度: {dim_z} × {dim_x} × {dim_y}")
    print(f"  对应总元素: {dim_z * dim_x * dim_y}")

    # 6. 如果需要采样
    if interval is not None:
        sampled_data = sample_with_interval(data_3d, interval)
        print("\n" + "=" * 70)
        print("保存采样数据")
        print("=" * 70)
        save_all_formats(
            sampled_data,
            output_dir,
            f'Seismic_0414_3D_sampled_interval{interval}'
        )
    else:
        print("\n不进行采样，只保存原始数据")

    # 7. 保存原始3D数据（始终保存）
    print("\n" + "=" * 70)
    print("保存原始3D数据")
    print("=" * 70)
    save_all_formats(
        data_3d,
        output_dir,
        'Seismic_0414_3D_original'
    )

    # 8. 输出汇总信息
    print("\n" + "=" * 70)
    print("处理完成！")
    print("=" * 70)
    print(f"原始SEGY读取后形状 (traces, samples): {data_2d.shape}")
    print(f"重塑后3D形状 (Time, Inline, Crossline): {data_3d.shape}")
    if interval is not None:
        print(f"采样后3D形状 (间隔={interval}): {sampled_data.shape}")
        print(f"数据点数: {data_3d.size} -> {sampled_data.size}")
        print(f"压缩比例: {sampled_data.size / data_3d.size * 100:.2f}%")
    print(f"\n输出目录: {output_dir}")
    print(f"  文件列表:")
    print(f"  - Seismic_0414_3D_original.npy/.bin/.mat/.dat/.txt/.csv (原始3D数据, 6种格式)")
    if interval is not None:
        print(f"  - Seismic_0414_3D_sampled_interval{interval}.npy/.bin/.mat/.dat/.txt/.csv (采样数据, 6种格式)")
    print("=" * 70)


if __name__ == "__main__":
    main()