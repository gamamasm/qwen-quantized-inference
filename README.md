# Qwen W4A16/W8A16 GEMM CUDA Fusion Kernel

针对 Qwen 系列大语言模型的 INT4/INT8 量化（仅权重）推理加速方案，实现 W4A16 和 W8A16 融合 CUDA 算子。

## 项目简介

本项目为 Qwen 系列模型（如 Qwen3-0.6B）提供了 **W4A16**（4-bit 权重 + FP16 激活）和 **W8A16**（8-bit 权重 + FP16 激活）量化推理支持。核心是自定义的 **CUDA 融合算子**，将反量化（dequantization）与矩阵乘法（GEMM）融合为单一 kernel，避免中间结果回写显存，减少内存带宽占用。

> **注意**：本项目的量化算法采用的是最基础的 per-group 对称量化 demo 实现，主要目的是展示 CUDA 融合算子的设计与集成流程。生产环境建议使用 GPTQ、AWQ 等成熟量化方案。

## 项目结构

```
├── w4a16_qwen/
│   ├── w4a16_qwen_gemm.cu      # W4A16 CUDA 融合算子
│   └── setup.py                 # 编译脚本
├── w8a16_qwen/
│   ├── w8a16_qwen_gemm.cu      # W8A16 CUDA 融合算子
│   └── setup.py                 # 编译脚本
├── quantize_qwen.py             # 模型量化脚本
├── test_qwen_quantize.py        # 量化模型推理测试
└── README.md
```

## CUDA 融合算子

### W4A16 Kernel (`w4a16_qwen_gemm.cu`)

- **权重量化**：将 FP16 权重按 group_size=128 分组，每组计算 scale = max(|w|) / 7.0，量化为 INT4（-8 ~ 7），两个 INT4 值打包为一个 uint8
- **融合计算**：每个 CUDA 线程处理一个输出元素，直接从 packed INT4 权重中解包、反量化、与 FP16 输入做乘加累加，无需中间反量化矩阵
- **输入适配**：支持任意维度输入（自动展平 batch*seq_len），适配 transformer 的 3D 张量 `(batch, seq_len, in_features)`

### W8A16 Kernel (`w8a16_qwen_gemm.cu`)

- **权重量化**：对称量化，scale = max(|w|) / 127，量化为 INT8（-128 ~ 127），无需打包
- **融合计算**：与 W4A16 类似，省去 INT4 解包的位操作，直接读取 INT8 权重做反量化乘加

### Kernel 参数

| 参数 | 形状 | 说明 |
|------|------|------|
| weight_int4 | (out_features, padded_in_features/2) | W4A16 打包后的 INT4 权重 |
| weight_int8 | (out_features, in_features) | W8A16 的 INT8 权重 |
| scale | (out_features, num_groups) | 每组的量化缩放因子 |
| input | (..., in_features) | FP16 输入激活 |
| bias | (out_features,) | 偏置（可选） |

## 快速开始

### 1. 编译 CUDA 算子

```bash
# W4A16
cd w4a16_qwen && python setup.py build_ext --inplace

# W8A16
cd w8a16_qwen && python setup.py build_ext --inplace
```

### 2. 量化模型

```bash
# W4A16 量化
python quantize_qwen.py --quant_type w4a16

# W8A16 量化
python quantize_qwen.py --quant_type w8a16
```

量化后的权重保存在 `model/Qwen3_06B_w4a16/` 或 `model/Qwen3_06B_w8a16/` 目录。

### 3. 推理测试

```bash
# W4A16 推理
python test_qwen_quantize.py --quant_type w4a16

# W8A16 推理
python test_qwen_quantize.py --quant_type w8a16
```

测试内容：10 条中文 prompt，输出平均推理时长和峰值显存占用。

## 依赖

- CUDA Toolkit 11.0+
- PyTorch 2.0+
- transformers
- GPU 计算能力 7.0+（如 RTX 3090/4090）

## 许可证

MIT
