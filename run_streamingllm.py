import argparse
import csv
import gc
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from transformers.cache_utils import DynamicCache

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"

PastKV = Tuple[Tuple[torch.Tensor, torch.Tensor], ...]

@dataclass
class MethodConfig:
    name: str
    budget_ratio: float = 1.0
    sink_size: int = 4
    window_size: int = 128

@dataclass
class Metrics:
    dataset: str
    method: str
    budget_ratio: float
    run_id: int
    input_length: int
    generated_tokens: int
    ppl: float
    ttft_sec: float
    tpot_ms: float
    throughput_tok_s: float
    peak_memory_mb: float
    kv_cache_memory_mb: float
    estimated_attention_flops_g: float

def parse_args():
    parser = argparse.ArgumentParser(description="StreamingLLM Full Experiment")
    parser.add_argument("--model_name", type=str, default="EleutherAI/pythia-70m")
    parser.add_argument("--datasets", nargs="+", default=["wikitext"])
    parser.add_argument("--input_length", type=int, default=2048)
    parser.add_argument("--ppl_eval_tokens", type=int, default=256)
    parser.add_argument("--generate_length", type=int, default=256)
    parser.add_argument("--budget_ratios", nargs="+", type=float, default=[0.2, 0.3, 0.5])
    parser.add_argument("--runs", type=int, default=2)
    parser.add_argument("--sink_size", type=int, default=4)
    parser.add_argument("--window_size", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="results_streamingllm")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="auto")
    return parser.parse_args()

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def dtype_from_arg(dtype_arg: str, device: torch.device):
    if dtype_arg == "float16": return torch.float16
    if dtype_arg == "bfloat16": return torch.bfloat16
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32

def synchronize(device: torch.device):
    if device.type == "cuda": torch.cuda.synchronize()

def reset_peak_memory(device: torch.device):
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

def get_peak_memory_mb(device: torch.device) -> float:
    if device.type != "cuda": return 0.0
    return torch.cuda.max_memory_allocated(device) / 1024 / 1024

def load_model_and_tokenizer(args, device: torch.device):
    dtype = dtype_from_arg(args.dtype, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=dtype, low_cpu_mem_usage=True, attn_implementation="eager"
    ).to(device).eval()
    return model, tokenizer

def iter_texts(dataset_name):
    if dataset_name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        for item in ds:
            txt = item["text"].strip()
            if txt:
                yield txt

def build_sample(tokenizer, name, total_tokens):
    buf = []
    for txt in iter_texts(name):
        buf.append(txt)
        ids = tokenizer("\n\n".join(buf), add_special_tokens=False, return_tensors="pt").input_ids
        if ids.shape[1] >= total_tokens:
            return ids[:, :total_tokens]
    raise RuntimeError("Not enough tokens")

def cache_to_tuple(past_key_values) -> PastKV:
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()
    if hasattr(past_key_values, "layers"):
        return tuple((layer.keys, layer.values) for layer in past_key_values.layers if layer.keys is not None)
    return past_key_values

def get_kv_cache_size_mb(past_key_values) -> float:
    past_tuple = cache_to_tuple(past_key_values)
    total_bytes = 0
    for k, v in past_tuple:
        total_bytes += k.numel() * k.element_size() + v.numel() * v.element_size()
    return total_bytes / 1024 / 1024

def streamingllm_compress_kv(past_key_values, sink_size: int, window_size: int, budget_ratio: float):
    past_tuple = cache_to_tuple(past_key_values)
    compressed = []
    for k, v in past_tuple:
        seq_len = k.shape[-2]
        max_cache = max(int(seq_len * budget_ratio), sink_size + window_size)
        recent_start = max(sink_size, seq_len - window_size)
        keep_indices = torch.cat([
            torch.arange(0, sink_size, device=k.device),
            torch.arange(recent_start, seq_len, device=k.device)
        ]).unique()
        if keep_indices.shape[0] > max_cache:
            keep_indices = keep_indices[-max_cache:]
        new_k = k.index_select(-2, keep_indices)
        new_v = v.index_select(-2, keep_indices)
        compressed.append((new_k.contiguous(), new_v.contiguous()))
    return DynamicCache(tuple(compressed))

@torch.inference_mode()
def prefill(model, input_ids, method: MethodConfig):
    outputs = model(input_ids, use_cache=True, return_dict=True)
    past_kv = outputs.past_key_values
    if method.name == "streamingllm":
        past_kv = streamingllm_compress_kv(
            past_kv, method.sink_size, method.window_size, method.budget_ratio
        )
    return outputs.logits, past_kv, get_kv_cache_size_mb(past_kv)

@torch.inference_mode()
def generate_full(model, input_ids, method: MethodConfig, gen_len: int, device: torch.device):
    reset_peak_memory(device)
    synchronize(device)
    start = time.perf_counter()

    logits, past_kv, kv_mb = prefill(model, input_ids, method)
    next_token = torch.argmax(logits[:, -1:], dim=-1)
    synchronize(device)
    ttft = time.perf_counter() - start

    generated = [next_token]
    cur_token = next_token
    seq_len = input_ids.shape[1]

    for _ in range(gen_len - 1):
        if method.name == "streamingllm":
            past_kv = streamingllm_compress_kv(
                past_kv, method.sink_size, method.window_size, method.budget_ratio
            )
        pos_ids = torch.tensor([[seq_len]], device=device)
        outputs = model(cur_token, past_key_values=past_kv, position_ids=pos_ids, use_cache=True)
        cur_token = torch.argmax(outputs.logits[:, -1:], dim=-1)
        generated.append(cur_token)
        seq_len += 1

    synchronize(device)
    total_time = time.perf_counter() - start
    tpot_ms = (total_time - ttft) / max(1, gen_len - 1) * 1000
    throughput = gen_len / total_time
    peak_mem = get_peak_memory_mb(device)

    return torch.cat(generated, dim=1), ttft, tpot_ms, throughput, peak_mem, kv_mb

# ===================== 已修复：PPL 计算函数 =====================
@torch.inference_mode()
def compute_ppl(model, input_ids, eval_len: int, method: MethodConfig, device: torch.device):
    ctx = input_ids[:, :-eval_len]
    tgt = input_ids[:, -eval_len:]
    logits, past_kv, _ = prefill(model, ctx, method)

    losses = []
    # 第一个token
    start_logits = logits[:, -1]
    first_tgt = tgt[:, 0]
    losses.append(F.cross_entropy(start_logits.float(), first_tgt))

    # 后续逐token
    for i in range(1, eval_len):
        token = tgt[:, i-1:i]
        pos = ctx.shape[1] + i - 1
        pos_ids = torch.tensor([[pos]], device=device)
        out = model(token, past_key_values=past_kv, position_ids=pos_ids, use_cache=True)
        past_kv = out.past_key_values

        if method.name == "streamingllm":
            past_kv = streamingllm_compress_kv(past_kv, method.sink_size, method.window_size, method.budget_ratio)

        step_logits = out.logits[:, -1]
        step_tgt = tgt[:, i]
        losses.append(F.cross_entropy(step_logits.float(), step_tgt))

    avg_loss = torch.stack(losses).mean().item()
    return math.exp(avg_loss)

def estimate_flops_g(model, input_len, gen_len, budget_ratio):
    layers = model.config.num_hidden_layers
    hidden = model.config.hidden_size
    prefill = 2 * layers * (input_len **2) * hidden
    decode = 2 * layers * gen_len * (input_len * budget_ratio) * hidden
    return (prefill + decode) / 1e9

def get_method_list(args) -> List[MethodConfig]:
    methods = [MethodConfig(name="baseline", budget_ratio=1.0)]
    for r in args.budget_ratios:
        methods.append(MethodConfig(name="streamingllm", budget_ratio=r))
    return methods

def save_results(rows: List[Metrics], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    data = [asdict(r) for r in rows]
    with open(os.path.join(output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    with open(os.path.join(output_dir, "results.csv"), "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

def print_summary(rows: List[Metrics]):
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[(r.dataset, r.method, r.budget_ratio)].append(r)

    print("\n===== 平均结果 =====")
    print("dataset,method,budget,ppl,ttft_sec,tpot_ms,throughput,peak_mem,kv_cache,flops")
    for key, vals in sorted(groups.items()):
        avg = lambda f: sum(getattr(v, f) for v in vals) / len(vals)
        print(f"{key[0]},{key[1]},{key[2]:.2f},"
              f"{avg('ppl'):.3f},{avg('ttft_sec'):.3f},{avg('tpot_ms'):.2f},"
              f"{avg('throughput_tok_s'):.2f},{avg('peak_memory_mb'):.1f},"
              f"{avg('kv_cache_memory_mb'):.1f},{avg('estimated_attention_flops_g'):.2f}")

def main():
    args = parse_args()
    set_seed(args.seed)
    random.seed(args.seed)
    device = get_device()
    print(f"设备: {device}")

    model, tokenizer = load_model_and_tokenizer(args, device)
    total = args.input_length + args.ppl_eval_tokens
    samples = {d: build_sample(tokenizer, d, total).to(device) for d in args.datasets}
    rows = []

    for dname, ids in samples.items():
        gen_input = ids[:, :args.input_length]
        ppl_input = ids
        for method in get_method_list(args):
            for run in range(args.runs):
                gc.collect()
                print(f"运行: {dname} | {method.name} | budget={method.budget_ratio} | run={run}")
                ppl = compute_ppl(model, ppl_input, args.ppl_eval_tokens, method, device)
                _, ttft, tpot, throughput, peak_mem, kv_mb = generate_full(
                    model, gen_input, method, args.generate_length, device
                )
                flops = estimate_flops_g(model, args.input_length, args.generate_length, method.budget_ratio)

                row = Metrics(
                    dataset=dname, method=method.name, budget_ratio=method.budget_ratio, run_id=run,
                    input_length=args.input_length, generated_tokens=args.generate_length,
                    ppl=ppl, ttft_sec=ttft, tpot_ms=tpot, throughput_tok_s=throughput,
                    peak_memory_mb=peak_mem, kv_cache_memory_mb=kv_mb, estimated_attention_flops_g=flops
                )
                rows.append(row)
                save_results(rows, args.output_dir)

    print_summary(rows)
    print(f"\n✅ 完成！结果保存在: {args.output_dir}")

if __name__ == "__main__":
    main()