import torch
import torch.nn as nn
import time
import os
import sys
import argparse
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'w4a16_qwen'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'w8a16_qwen'))

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

os.environ['LD_LIBRARY_PATH'] = '/root/anaconda3/envs/pytorch_2.5.1/lib/python3.10/site-packages/torch/lib:/usr/local/cuda/lib64' + (':' + os.environ['LD_LIBRARY_PATH'] if 'LD_LIBRARY_PATH' in os.environ else '')

try:
    import w4a16_qwen_gemm
    HAS_W4A16_KERNEL = True
except ImportError:
    HAS_W4A16_KERNEL = False

try:
    import w8a16_qwen_gemm
    HAS_W8A16_KERNEL = True
except ImportError:
    HAS_W8A16_KERNEL = False

if not HAS_W4A16_KERNEL and not HAS_W8A16_KERNEL:
    print("Warning: No CUDA kernel compiled")


# ==================== W4A16 Linear ====================

class W4A16Linear(nn.Module):
    def __init__(self, in_features, out_features, group_size=128, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.num_groups = in_features // group_size
        self.padded_in_features = (in_features + 1) // 2 * 2

        self.register_buffer('weight_int4', torch.zeros(out_features, self.padded_in_features // 2, dtype=torch.uint8))
        self.register_buffer('scale', torch.ones(out_features, self.num_groups, dtype=torch.float16))

        if bias:
            self.register_buffer('bias', torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

    def forward(self, x):
        original_shape = x.shape
        if x.dim() > 2:
            x = x.view(-1, self.in_features)

        out = w4a16_qwen_gemm.w4a16_qwen_gemm(
            self.weight_int4,
            self.scale,
            x.half(),
            self.bias if self.bias is not None else torch.tensor([], dtype=torch.float16),
            self.in_features
        )

        new_shape = original_shape[:-1] + (self.out_features,)
        return out.view(new_shape)


# ==================== W8A16 Linear ====================

class W8A16Linear(nn.Module):
    def __init__(self, in_features, out_features, group_size=128, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size
        self.num_groups = in_features // group_size

        self.register_buffer('weight_int8', torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('scale', torch.ones(out_features, self.num_groups, dtype=torch.float16))

        if bias:
            self.register_buffer('bias', torch.zeros(out_features, dtype=torch.float16))
        else:
            self.bias = None

    def forward(self, x):
        original_shape = x.shape
        if x.dim() > 2:
            x = x.view(-1, self.in_features)

        out = w8a16_qwen_gemm.w8a16_qwen_gemm(
            self.weight_int8,
            self.scale,
            x.half(),
            self.bias if self.bias is not None else torch.tensor([], dtype=torch.float16),
            self.in_features
        )

        new_shape = original_shape[:-1] + (self.out_features,)
        return out.view(new_shape)


# ==================== 替换线形层 ====================

def replace_linear_with_quantized(model, quant_type="w4a16", group_size=128):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            child_name = name.split('.')[-1] if '.' in name else name

            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model

            if quant_type == "w8a16":
                quantized_linear = W8A16Linear(
                    module.in_features, module.out_features,
                    group_size=group_size, bias=module.bias is not None
                )
            else:
                quantized_linear = W4A16Linear(
                    module.in_features, module.out_features,
                    group_size=group_size, bias=module.bias is not None
                )

            setattr(parent, child_name, quantized_linear)
        else:
            replace_linear_with_quantized(module, quant_type, group_size)
    return model


# ==================== 加载模型 ====================

def load_quantized_model(model_path, tokenizer_path, quant_type="w4a16"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 在 CPU 上加载模型结构
    original_model_path = "model/Qwen3_06B"
    config = AutoConfig.from_pretrained(original_model_path)
    with torch.device("cpu"):
        model = AutoModelForCausalLM.from_config(config, dtype=torch.float16)

    # 替换线形层为量化层（仍在 CPU）
    model = replace_linear_with_quantized(model, quant_type=quant_type)

    # 在 CPU 上加载量化后的权重
    state_dict = torch.load(os.path.join(model_path, "quantized_model.pt"), weights_only=True, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)

    # 在 CPU 上加载非量化层的原始权重（layernorm, embed_tokens 等）
    from safetensors.torch import load_file
    original_state = load_file(os.path.join(original_model_path, "model.safetensors"))
    model_state = model.state_dict()
    for key, value in original_state.items():
        if key in model_state and key not in state_dict:
            model_state[key] = value
    model.load_state_dict(model_state, strict=False)

    # 所有权重加载完成后再移到 GPU
    model = model.to(device)
    model.eval()

    return model, tokenizer


# ==================== 测试数据 ====================

test_prompts = [
    "介绍一下马斯克。",
    "什么是人工智能？",
    "Python的主要优点是什么？",
    "解释一下量子计算的基本原理。",
    "中国的首都是哪里？",
    "太阳系有哪些行星？",
    "什么是机器学习？",
    "简述HTTP和HTTPS的区别。",
    "什么是深度学习？",
    "解释什么是神经网络。",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant_type", type=str, default="w8a16", choices=["w4a16", "w8a16"])
    args = parser.parse_args()

    model_path = f"/mnt/d/python/GEMM/model/Qwen3_06B_{args.quant_type}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Quantization type: {args.quant_type}")

    print(f"Loading quantized model from {model_path}...")
    model, tokenizer = load_quantized_model(model_path, model_path, quant_type=args.quant_type)
    print("Model loaded successfully!")

    results = []
    total_time = 0
    peak_memory = 0

    print("开始测试推理...")
    print("=" * 60)

    for i, prompt in enumerate(test_prompts):
        messages = [
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        torch.cuda.synchronize()
        start_time = time.time()

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=32768
        )

        torch.cuda.synchronize()
        end_time = time.time()

        inference_time = end_time - start_time
        total_time += inference_time

        current_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
        if current_memory > peak_memory:
            peak_memory = current_memory

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()

        try:
            index = len(output_ids) - output_ids[::-1].index(151668)
        except ValueError:
            index = 0

        content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

        results.append((prompt, content, inference_time))
        print(f"完成第 {i+1}/{len(test_prompts)} 条测试, 耗时: {inference_time:.2f}s")

    avg_time = total_time / len(test_prompts)

    print("\n" + "=" * 60)
    print("所有输入和输出:")
    print("=" * 60)

    for i, (prompt, content, inference_time) in enumerate(results):
        print(f"\n--- 第 {i+1} 条 ---")
        print(f"输入: {prompt}")
        print(f"输出: {content}")
        print(f"推理时长: {inference_time:.2f}s")

    print("\n" + "=" * 60)
    print("测试统计:")
    print("=" * 60)
    print(f"量化类型: {args.quant_type}")
    print(f"测试条数: {len(test_prompts)}")
    print(f"总推理时长: {total_time:.2f}s")
    print(f"平均推理时长: {avg_time:.2f}s")
    print(f"峰值显存占用: {peak_memory:.2f}MB")


if __name__ == "__main__":
    main()
