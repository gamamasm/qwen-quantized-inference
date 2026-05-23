import torch
import torch.nn as nn
import os
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer


# ==================== W4A16 ====================

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
        dev = x.device

        w = self.weight_int4.to(dev)
        s = self.scale.to(dev)

        w_low = (w & 0x0F).float()
        w_high = ((w >> 4) & 0x0F).float()

        w_interleaved = torch.stack([w_low, w_high], dim=2)
        w_expanded = w_interleaved.view(self.out_features, self.padded_in_features)

        if self.in_features < self.padded_in_features:
            w_expanded = w_expanded[:, :self.in_features]

        w_expanded = (w_expanded - 8.0) * s.repeat(1, self.group_size)
        w_expanded = w_expanded.half()

        b = self.bias
        if b is not None:
            b = b.to(dev)

        return nn.functional.linear(x, w_expanded, b)


def quantize_weight_to_int4(weight_fp16, group_size=128):
    out_features, in_features = weight_fp16.shape
    num_groups = in_features // group_size

    weight_groups = weight_fp16.view(out_features, num_groups, group_size)
    weight_max = weight_groups.abs().max(dim=2)[0]
    scale = weight_max / 7.0

    weight_int4 = torch.round(weight_groups / scale.unsqueeze(2))
    weight_int4 = (weight_int4 + 8).clamp(0, 15).to(torch.uint8)

    padded_in_features = (in_features + 1) // 2 * 2
    if in_features < padded_in_features:
        padding = torch.zeros(out_features, padded_in_features - in_features, dtype=torch.uint8)
        weight_int4 = torch.cat([weight_int4.view(out_features, -1), padding], dim=1)

    weight_flat = weight_int4.view(out_features, -1)

    weight_even = weight_flat[:, 0::2]
    weight_odd = weight_flat[:, 1::2]

    weight_packed = weight_even | (weight_odd << 4)

    return weight_packed, scale


# ==================== W8A16 ====================

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
        dev = x.device

        w = self.weight_int8.to(dev).float()
        s = self.scale.to(dev)

        # 反量化：weight_fp16 = weight_int8 * scale
        # scale shape: (out_features, num_groups), 需要repeat到 (out_features, in_features)
        w_expanded = w * s.repeat(1, self.group_size)
        w_expanded = w_expanded.half()

        b = self.bias
        if b is not None:
            b = b.to(dev)

        return nn.functional.linear(x, w_expanded, b)


def quantize_weight_to_int8(weight_fp16, group_size=128):
    out_features, in_features = weight_fp16.shape
    num_groups = in_features // group_size

    weight_groups = weight_fp16.view(out_features, num_groups, group_size)
    weight_max = weight_groups.abs().max(dim=2)[0]
    # 对称量化：scale = max / 127
    scale = weight_max / 127.0

    weight_int8 = torch.round(weight_groups / scale.unsqueeze(2))
    weight_int8 = weight_int8.clamp(-128, 127).to(torch.int8)

    # 展平为 2D (out_features, in_features)
    weight_int8 = weight_int8.view(out_features, in_features)

    return weight_int8, scale


# ==================== 量化入口 ====================

def quantize_model(model, quant_type="w4a16", group_size=128):
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            parent_name = name.rsplit('.', 1)[0] if '.' in name else ''
            child_name = name.split('.')[-1] if '.' in name else name

            if parent_name:
                parent = model.get_submodule(parent_name)
            else:
                parent = model

            if quant_type == "w8a16":
                weight_int8, scale = quantize_weight_to_int8(module.weight.data, group_size=group_size)
                quantized_linear = W8A16Linear(
                    module.in_features, module.out_features,
                    group_size=group_size, bias=module.bias is not None
                )
                quantized_linear.weight_int8 = weight_int8
                quantized_linear.scale = scale
            else:
                weight_int4, scale = quantize_weight_to_int4(module.weight.data, group_size=group_size)
                quantized_linear = W4A16Linear(
                    module.in_features, module.out_features,
                    group_size=group_size, bias=module.bias is not None
                )
                quantized_linear.weight_int4 = weight_int4
                quantized_linear.scale = scale

            if module.bias is not None:
                quantized_linear.bias = module.bias.data.clone()

            setattr(parent, child_name, quantized_linear)
            print(f"Quantized ({quant_type}): {name}")
        else:
            quantize_model(module, quant_type, group_size)

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant_type", type=str, default="w8a16", choices=["w4a16", "w8a16"])
    parser.add_argument("--model_name", type=str, default="model/Qwen3_06B")
    args = parser.parse_args()

    output_dir = f"/mnt/d/python/GEMM/model/Qwen3_06B_{args.quant_type}"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Quantization type: {args.quant_type}")

    print(f"Loading model from {args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    os.makedirs(output_dir, exist_ok=True)

    tokenizer.save_pretrained(output_dir)
    print(f"Tokenizer saved to {output_dir}")

    print("Quantizing model...")
    model = quantize_model(model, quant_type=args.quant_type)
    model.eval()

    print(f"Saving quantized model to {output_dir}...")
    torch.save(model.state_dict(), os.path.join(output_dir, "quantized_model.pt"))
    print("Quantization complete!")



if __name__ == "__main__":
    main()
