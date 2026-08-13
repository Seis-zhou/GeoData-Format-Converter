# GeoData-Format-Converter
Python toolkit for processing SEGY and MAT geophysical data with multi-format conversion and resampling capabilities.

工作流程 / Workflow

步骤一 / Step 1: SEGY 处理 / SEGY Processing
读取 SEGY 文件，转置数据矩阵，保存为 6 种格式（NPY/BIN/MAT/DAT/TXT/CSV）
Read SEGY files, transpose data matrix, save to 6 formats (NPY/BIN/MAT/DAT/TXT/CSV)

步骤二 / Step 2: MAT 重采样 / MAT Resampling
加载 MAT 文件，10 倍降采样（双线性插值），保存为 6 种格式
Load MAT files, 10x downsampling (bilinear interpolation), save to 6 formats

步骤三 / Step 3: 格式互转 / Format Conversion
各格式相互转换，生成多格式对比图
Convert between all formats, generate comparison figures
