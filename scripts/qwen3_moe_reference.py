"""Greedy reference tokens of a block-FP8 Qwen3 MoE checkpoint, computed on the CPU.

The forward is written out in PyTorch from the checkpoint's tensors, so it does not depend on
how a framework fuses or quantizes the experts:

- Every projection's E4M3 codes are multiplied by their 128x128 block's `weight_scale_inv`.
- `--experts w8a8` quantizes each expert projection's input at run time, one E4M3 scale per group
  of 128 inputs (amax / 448), as vLLM's and the device's block-FP8 linears do; `w8a16` multiplies
  the dequantized weights with the BF16 input, as Transformers does.
- `--transformers` runs Transformers' `Qwen3MoeForCausalLM` on the dequantized weights instead,
  whose greedy tokens the `w8a16` ones match.

Usage: python scripts/qwen3_moe_reference.py /workspace/models/Qwen3-30B-A3B-FP8 --experts w8a8
"""

import argparse
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

# "Instruct: Given a web search query, retrieve relevant passages that answer the query
# Query:What is the capital of China?" in Qwen's tokens: the engine tests' shared prompt.
PROMPT = [
    641, 1235, 25, 16246, 264, 3482, 2711, 3239, 11, 17179, 9760, 46769, 429, 4226, 279, 3239,
    11361, 25, 3838, 374, 279, 6722, 315, 5616, 30,
]
E4M3_MAX = 448.0
BLOCK = 128


def tensors(directory: Path):
    index = json.loads((directory / "model.safetensors.index.json").read_text())["weight_map"]
    files = {}
    for name, file in index.items():
        files.setdefault(file, []).append(name)
    for file, names in files.items():
        with safe_open(directory / file, "pt") as handle:
            for name in names:
                yield name, handle.get_tensor(name)


def dequantize(codes: torch.Tensor, scale_inv: torch.Tensor) -> torch.Tensor:
    """BF16 values of `codes` [out, in] under one scale per 128x128 block."""
    rows, columns = codes.shape
    scales = scale_inv.float().repeat_interleave(BLOCK, 0)[:rows].repeat_interleave(BLOCK, 1)
    return (codes.float() * scales[:, :columns]).to(torch.bfloat16)


def quantize_groups(x: torch.Tensor) -> torch.Tensor:
    """`x` [rows, in] through E4M3 with one scale per row's group of 128 inputs, back to FP32."""
    rows, columns = x.shape
    groups = x.float().view(rows, columns // BLOCK, BLOCK)
    scale = groups.abs().amax(-1, keepdim=True).clamp(min=1e-10) * (1.0 / E4M3_MAX)
    codes = (groups / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return (codes.float() * scale).view(rows, columns)


def load(directory: Path, experts: bool):
    """Each weight in BF16, its codes dequantized; an expert's codes and scales are kept as they are
    unless `experts`, which dequantizes them too."""
    raw = dict(tensors(directory))
    weights = {}
    for name in list(raw):
        if name.endswith("weight_scale_inv"):
            continue
        value = raw.pop(name)
        scale = raw.get(name + "_scale_inv")
        if scale is None:
            weights[name] = value.to(torch.bfloat16)
        elif ".experts." in name and not experts:
            weights[name] = (value, scale)
        else:
            weights[name] = dequantize(value, scale)
    return weights


def rms_norm(x, weight, eps):
    variance = x.float().pow(2).mean(-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps)).to(torch.bfloat16) * weight


def rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def linear(x, weight, w8a8=False):
    if w8a8:
        return (quantize_groups(x) @ weight.float().T).to(torch.bfloat16)
    return x @ weight.T


class Reference:
    def __init__(self, directory: Path, experts: str):
        self.config = json.loads((directory / "config.json").read_text())
        self.w = load(directory, experts=False)
        self.w8a8 = experts == "w8a8"

    def forward(self, ids):
        c, w = self.config, self.w
        heads, kv, dim = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        eps, theta = c["rms_norm_eps"], c["rope_theta"]
        n = len(ids)
        x = w["model.embed_tokens.weight"][torch.tensor(ids)]
        inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        freqs = torch.outer(torch.arange(n, dtype=torch.float32), inv)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos().to(torch.bfloat16), emb.sin().to(torch.bfloat16)
        mask = torch.full((n, n), float("-inf")).triu(1)
        for layer in range(c["num_hidden_layers"]):
            p = f"model.layers.{layer}."
            h = rms_norm(x, w[p + "input_layernorm.weight"], eps)
            q = linear(h, w[p + "self_attn.q_proj.weight"]).view(n, heads, dim)
            k = linear(h, w[p + "self_attn.k_proj.weight"]).view(n, kv, dim)
            v = linear(h, w[p + "self_attn.v_proj.weight"]).view(n, kv, dim)
            q = rms_norm(q, w[p + "self_attn.q_norm.weight"], eps)
            k = rms_norm(k, w[p + "self_attn.k_norm.weight"], eps)
            q = q * cos[:, None] + rotate_half(q) * sin[:, None]
            k = k * cos[:, None] + rotate_half(k) * sin[:, None]
            k = k.repeat_interleave(heads // kv, dim=1)
            v = v.repeat_interleave(heads // kv, dim=1)
            scores = torch.einsum("qhd,khd->hqk", q.float(), k.float()) / math.sqrt(dim) + mask
            probs = scores.softmax(-1).to(torch.bfloat16)
            attended = torch.einsum("hqk,khd->qhd", probs, v).reshape(n, heads * dim)
            x = x + linear(attended, w[p + "self_attn.o_proj.weight"])
            h = rms_norm(x, w[p + "post_attention_layernorm.weight"], eps)
            logits = h @ w[p + "mlp.gate.weight"].T
            routing = logits.float().softmax(-1)
            top, chosen = routing.topk(c["num_experts_per_tok"], dim=-1)
            top = (top / top.sum(-1, keepdim=True)).to(torch.bfloat16)
            # Each selection's row, summed in FP32 in choice order and rounded once.
            routed = torch.zeros(n, chosen.shape[1], h.shape[1], dtype=torch.bfloat16)
            for expert in chosen.unique().tolist():
                e = f"{p}mlp.experts.{expert}."
                gate_w, up_w, down_w = (
                    dequantize(*w[e + name + ".weight"])
                    for name in ("gate_proj", "up_proj", "down_proj")
                )
                rows, slots = (chosen == expert).nonzero(as_tuple=True)
                x_e = h[rows]
                gate = linear(x_e, gate_w, self.w8a8)
                up = linear(x_e, up_w, self.w8a8)
                product = torch.nn.functional.silu(gate) * up
                routed[rows, slots] = linear(product, down_w, self.w8a8)
            out = (routed.float() * top.float()[..., None]).sum(1).to(torch.bfloat16)
            x = x + out
        x = rms_norm(x, w["model.norm.weight"], eps)
        return x[-1] @ w["lm_head.weight"].T

    def greedy(self, prompt, count):
        """The greedy continuation, and each step's lead of its token over the runner-up."""
        ids, margins = list(prompt), []
        for _ in range(count):
            top = self.forward(ids).float().topk(2)
            ids.append(int(top.indices[0]))
            margins.append(round(float(top.values[0] - top.values[1]), 3))
        return ids[len(prompt) :], margins


def transformers_tokens(directory: Path, count):
    from transformers import AutoConfig, Qwen3MoeForCausalLM

    config = AutoConfig.from_pretrained(directory)
    config.quantization_config = None
    del config.quantization_config
    with torch.device("meta"):
        model = Qwen3MoeForCausalLM(config)
    model = model.to_empty(device="cpu").to(torch.bfloat16)
    # `to_empty` leaves the buffers no state dict holds, the rotary frequencies, uninitialized.
    model.model.rotary_emb = type(model.model.rotary_emb)(config=config)
    state = model.state_dict()
    written = set()
    scales = {n: t for n, t in tensors(directory) if n.endswith("weight_scale_inv")}
    with torch.no_grad():
        for name, value in tensors(directory):
            if name.endswith("weight_scale_inv"):
                continue
            scale = scales.get(name + "_scale_inv")
            value = value.to(torch.bfloat16) if scale is None else dequantize(value, scale)
            if name in state:
                state[name].copy_(value)
                written.add(name)
                continue
            # An expert's projection lands in its layer's fused gate/up or down tensor.
            prefix, expert, projection = name.rsplit(".", 3)[0], *name.rsplit(".", 3)[1:3]
            fused = f"{prefix}.gate_up_proj" if projection != "down_proj" else f"{prefix}.down_proj"
            target = state[fused][int(expert)]
            if projection == "down_proj":
                target.copy_(value.view(target.shape))
            else:
                half = target.shape[0] // 2
                rows = slice(0, half) if projection == "gate_proj" else slice(half, None)
                target[rows].copy_(value)
            written.add(fused)
    missing = set(state) - written
    assert not missing, sorted(missing)[:8]
    model.eval()
    with torch.no_grad():
        out = model.generate(torch.tensor([PROMPT]), max_new_tokens=count, do_sample=False)
    return out[0, len(PROMPT) :].tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--experts", choices=["w8a8", "w8a16"], default="w8a8")
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--transformers", action="store_true")
    parser.add_argument("--text", help="a prompt to tokenize in place of the shared one")
    args = parser.parse_args()
    torch.set_num_threads(8)
    if args.transformers:
        hf = transformers_tokens(args.checkpoint, args.tokens)
        print(f"transformers w8a16: {hf}", flush=True)
        return
    prompt = PROMPT
    if args.text is not None:
        from transformers import AutoTokenizer

        prompt = AutoTokenizer.from_pretrained(args.checkpoint)(args.text)["input_ids"]
        print(f"prompt: {prompt}", flush=True)
    reference = Reference(args.checkpoint, args.experts)
    with torch.no_grad():
        tokens, margins = reference.greedy(prompt, args.tokens)
    print(f"{args.experts}: {tokens} margins {margins}", flush=True)


if __name__ == "__main__":
    main()
