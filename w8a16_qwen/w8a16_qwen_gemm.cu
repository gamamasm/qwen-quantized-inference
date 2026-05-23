#include <torch/extension.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <iostream>

// W8A16 GEMM CUDA kernel - INT8权重，FP16激活
// 与w4a16结构一致，区别：权重为INT8，无需打包，1字节=1个权重值
__global__ void w8a16_qwen_gemm_kernel(
    const int8_t* __restrict__ weight_int8,   // INT8权重，每个元素1字节
    const at::Half* __restrict__ scale,        // 量化缩放因子
    const at::Half* __restrict__ input,        // FP16输入
    const at::Half* __restrict__ bias,         // 偏置（可选）
    at::Half* __restrict__ output,             // FP16输出
    int batch_size,
    int in_features,
    int out_features,
    int num_groups,
    int group_size
) {
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= batch_size || col >= out_features) return;

    float sum = 0.0f;

    if (bias != nullptr) {
        sum = __half2float(bias[col]);
    }

    // INT8无需padded_in_features，每元素占1字节
    for (int g = 0; g < num_groups; ++g) {
        const int scale_offset = col * num_groups + g;
        at::Half scale_val = scale[scale_offset];

        for (int i = 0; i < group_size; ++i) {
            const int col_idx = g * group_size + i;
            if (col_idx >= in_features) break;

            // INT8权重直接读取，无需位操作解包
            const int weight_offset = col * in_features + col_idx;
            const int input_offset = row * in_features + col_idx;

            int8_t w_int8 = weight_int8[weight_offset];

            float w_fp = static_cast<float>(w_int8) * __half2float(scale_val);
            float x_fp = __half2float(input[input_offset]);

            sum += w_fp * x_fp;
        }
    }

    output[row * out_features + col] = __float2half(sum);
}

torch::Tensor w8a16_qwen_gemm(
    torch::Tensor weight_int8,
    torch::Tensor scale,
    torch::Tensor input,
    torch::Tensor bias,
    int in_features
) {
    // 展平前N-1维为batch_size
    auto input_sizes = input.sizes().vec();
    int batch_size = 1;
    for (int i = 0; i < (int)input_sizes.size() - 1; ++i) {
        batch_size *= input_sizes[i];
    }

    auto input_2d = input.contiguous().reshape({batch_size, in_features});

    const int out_features = weight_int8.size(0);
    const int num_groups = scale.size(1);
    const int group_size = in_features / num_groups;

    auto output_2d = torch::empty({batch_size, out_features}, input.options());

    const dim3 block(16, 16);
    const dim3 grid(
        (out_features + block.x - 1) / block.x,
        (batch_size + block.y - 1) / block.y
    );

    w8a16_qwen_gemm_kernel<<<grid, block>>>(
        weight_int8.data_ptr<int8_t>(),
        scale.data_ptr<at::Half>(),
        input_2d.data_ptr<at::Half>(),
        bias.defined() ? bias.data_ptr<at::Half>() : nullptr,
        output_2d.data_ptr<at::Half>(),
        batch_size,
        in_features,
        out_features,
        num_groups,
        group_size
    );

    auto output_sizes = input_sizes;
    output_sizes.back() = out_features;
    return output_2d.reshape(output_sizes);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("w8a16_qwen_gemm", &w8a16_qwen_gemm, "W8A16 Qwen GEMM kernel (INT8 weights)");
}
