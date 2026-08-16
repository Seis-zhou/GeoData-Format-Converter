# GeoData-Format-Converter
Python toolkit for processing SEGY and MAT geophysical data with multi-format conversion and resampling capabilities.

**2D_data**

2D Seismic Data Processing Tools / 2D地震数据处理工具

Steps / 步骤

步骤一 / Step 1: SEGY 处理 / SEGY Processing

读取 SEGY 文件，转置数据矩阵，保存为 6 种格式（NPY/BIN/MAT/DAT/TXT/CSV）
Read SEGY files, transpose data matrix, save to 6 formats (NPY/BIN/MAT/DAT/TXT/CSV)

步骤二 / Step 2: MAT 重采样 / MAT Resampling

加载 MAT 文件，10 倍降采样（双线性插值），保存为 6 种格式
Load MAT files, 10x downsampling (bilinear interpolation), save to 6 formats

步骤三 / Step 3: 格式互转 / Format Conversion

各格式相互转换，生成多格式对比图
Convert between all formats, generate comparison figures



**3D_data**

3D Seismic Data Processing Tools / 3D地震数据处理工具

This repository contains tools for processing and visualizing 3D seismic data. / 本仓库包含处理和可视化3D地震数据的工具。

Steps / 步骤

Step1: SEGY to 3D Format Conversion / SEGY转3D格式

Convert SEGY file to multiple 3D formats (NPY, BIN, MAT, DAT, TXT, CSV). / 将SEGY文件转换为多种3D格式（NPY、BIN、MAT、DAT、TXT、CSV）。

Step2: 3D Data Visualization / 3D数据可视化

Visualize 3D seismic data with three-direction slices, reference lines, and customizable colormaps. / 使用三方向切片、基准线和可定制色标可视化3D地震数据。

Step3: Format Conversion & Comparison / 格式转换与对比

Convert between all 6 formats and generate comparison figures. / 在6种格式之间相互转换并生成对比图。
