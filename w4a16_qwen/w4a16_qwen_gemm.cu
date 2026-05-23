#include <torch/extension.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <iostream>

__global__ void w4a16_qwen_gemm_kernel(
    const uint8_t* __restrict__ weight_int4,
    const at::Half* __restrict__ scale,
    const at::Half* __restrict__ input,
    const at::Half* __restrict__ bias,
    at::Half* __restrict__ output,
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

    const int padded_in_features = (in_features + 1) / 2 * 2;

    for (int g = 0; g < num_groups; ++g) {
        const int scale_offset = col * num_groups + g;
        at::Half scale_val = scale[scale_offset];

        for (int i = 0; i < group_size; ++i) {
            const int col_idx = g * group_size + i;
            if (col_idx >= in_features) break;

            const int packed_idx = col_idx / 2;
            const int bit_shift = (col_idx % 2) * 4;

            const int weight_offset = col * ((padded_in_features + 1) / 2) + packed_idx;
            const int input_offset = row * in_features + col_idx;

            uint8_t packed = weight_int4[weight_offset];
            uint8_t w_int4 = (packed >> bit_shift) & 0x0F;

            float w_fp = (static_cast<float>(w_int4) - 8.0f) * __half2float(scale_val);
            float x_fp = __half2float(input[input_offset]);

            sum += w_fp * x_fp;
        }
    }

    output[row * out_features + col] = __float2half(sum);
}

torch::Tensor w4a16_qwen_gemm(
    torch::Tensor weight_int4,
    torch::Tensor scale,
    torch::Tensor input,
    torch::Tensor bias,
    int in_features
) {
    const int batch_size = input.size(0);
    const int out_features = weight_int4.size(0);
    const int num_groups = scale.size(1);
    const int group_size = in_features / num_groups;

    auto output = torch::empty({batch_size, out_features}, input.options());

    const dim3 block(16, 16);
    const dim3 grid(
        (out_features + block.x - 1) / block.x,
        (batch_size + block.y - 1) / block.y
    );

    w4a16_qwen_gemm_kernel<<<grid, block>>>(
        weight_int4.data_ptr<uint8_t>(),
        scale.data_ptr<at::Half>(),
        input.data_ptr<at::Half>(),
        bias.defined() ? bias.data_ptr<at::Half>() : nullptr,
        output.data_ptr<at::Half>(),
        batch_size,
        in_features,
        out_features,
        num_groups,
        group_size
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("w4a16_qwen_gemm", &w4a16_qwen_gemm, "W4A16 Qwen GEMM kernel");
}
