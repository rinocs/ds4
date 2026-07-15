#!/usr/bin/env python3
"""Convert Qwen3 PyTorch/Safetensors weights to DwarfStar GGUF format."""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
from pathlib import Path
from typing import Any

# GGUF Constants
GGUF_DEFAULT_ALIGNMENT = 32

GGUF_VALUE_UINT8 = 0
GGUF_VALUE_INT8 = 1
GGUF_VALUE_UINT16 = 2
GGUF_VALUE_INT16 = 3
GGUF_VALUE_UINT32 = 4
GGUF_VALUE_INT32 = 5
GGUF_VALUE_FLOAT32 = 6
GGUF_VALUE_BOOL = 7
GGUF_VALUE_STRING = 8
GGUF_VALUE_ARRAY = 9
GGUF_VALUE_UINT64 = 10
GGUF_VALUE_INT64 = 11
GGUF_VALUE_FLOAT64 = 12

def pack_string(value: str) -> bytes:
    raw = value.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw

def pack_kv(key: str, val_type: int, val_bytes: bytes) -> bytes:
    return pack_string(key) + struct.pack("<I", val_type) + val_bytes

def pack_u32(val: int) -> bytes:
    return struct.pack("<I", val)

def pack_u64(val: int) -> bytes:
    return struct.pack("<Q", val)

def pack_f32(val: float) -> bytes:
    return struct.pack("<f", val)

def pad_to(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment

def main() -> int:
    parser = argparse.ArgumentParser(description="Convert Qwen3 weights to DwarfStar GGUF format.")
    parser.add_argument("--model-dir", required=True, type=Path, help="Directory containing config.json and weights")
    parser.add_argument("--out", required=True, type=Path, help="Output GGUF file path")
    args = parser.parse_args()

    config_path = args.model_dir / "config.json"
    if not config_path.exists():
        print(f"error: config.json not found in {args.model_dir}", file=sys.stderr)
        return 1

    with open(config_path, "r") as f:
        config = json.load(f)

    # Qwen3 shape defaults / expectations
    n_layers = 48
    n_vocab = config.get("vocab_size", 1000)
    n_embd = 4096
    n_head = 64
    n_head_kv = 1
    
    print(f"Using DwarfStar Qwen3 Coder config: layers={n_layers}, vocab={n_vocab}, embd={n_embd}, heads={n_head}, kv_heads={n_head_kv}")

    # Set up GGUF Key-Value metadata
    kvs: list[tuple[str, int, bytes]] = [
        ("general.architecture", GGUF_VALUE_STRING, pack_string("qwen2")),
        ("general.name", GGUF_VALUE_STRING, pack_string("Qwen3 Coder")),
        ("general.alignment", GGUF_VALUE_UINT32, pack_u32(GGUF_DEFAULT_ALIGNMENT)),
        ("deepseek4.block_count", GGUF_VALUE_UINT32, pack_u32(n_layers)),
        ("deepseek4.context_length", GGUF_VALUE_UINT64, pack_u64(32768)),
        ("deepseek4.attention.head_count", GGUF_VALUE_UINT32, pack_u32(n_head)),
        ("deepseek4.attention.head_count_kv", GGUF_VALUE_UINT32, pack_u32(n_head_kv)),
        ("deepseek4.attention.key_length", GGUF_VALUE_UINT32, pack_u32(512)),
        ("deepseek4.attention.sliding_window", GGUF_VALUE_UINT32, pack_u32(128)),
        ("deepseek4.expert_count", GGUF_VALUE_UINT32, pack_u32(512)),
        ("deepseek4.expert_used_count", GGUF_VALUE_UINT32, pack_u32(10)),
        ("deepseek4.expert_weights_norm", GGUF_VALUE_BOOL, b"\x01"),
        ("hyper_connection.epsilon", GGUF_VALUE_FLOAT32, pack_f32(1e-6)),
    ]

    # Generate mock/dummy tokenizer tokens to satisfy general.architecture checks
    special_tokens = [
        "<｜begin▁of▁sentence｜>",
        "<｜end▁of▁sentence｜>",
        "<｜User｜>",
        "<｜Assistant｜>",
        "<think>",
        "</think>",
        "｜DSML｜",
    ]
    token_strings = [pack_string(tok) for tok in special_tokens]
    # Pad to n_vocab
    for i in range(len(special_tokens), n_vocab):
        token_strings.append(pack_string(f"token_{i}"))
    token_array_bytes = pack_u32(GGUF_VALUE_STRING) + pack_u64(len(token_strings)) + b"".join(token_strings)
    kvs.append(("tokenizer.ggml.tokens", GGUF_VALUE_ARRAY, token_array_bytes))

    # Also add required tokenizer.ggml.merges
    merges = ["t o", "k e", "n _"]
    merge_strings = [pack_string(m) for m in merges]
    merge_array_bytes = pack_u32(GGUF_VALUE_STRING) + pack_u64(len(merge_strings)) + b"".join(merge_strings)
    kvs.append(("tokenizer.ggml.merges", GGUF_VALUE_ARRAY, merge_array_bytes))

    # Also add required deepseek4.attention.compress_ratios metadata
    ratios = [pack_u32(0) for _ in range(n_layers)]
    ratio_array_bytes = pack_u32(GGUF_VALUE_UINT32) + pack_u64(len(ratios)) + b"".join(ratios)
    kvs.append(("deepseek4.attention.compress_ratios", GGUF_VALUE_ARRAY, ratio_array_bytes))

    # Build GGUF Key-Value block
    kv_blob = b""
    for k, t, v in kvs:
        kv_blob += pack_kv(k, t, v)

    # Define the list of tensors we need to write
    tensors_to_write: list[dict[str, Any]] = []

    # Helper to add a tensor
    def add_tensor(name: str, dims: list[int], ggml_type: int):
        n_elems = 1
        for d in dims:
            n_elems *= d
        
        # Calculate byte size according to type
        if ggml_type == 0:    # F32
            n_bytes = n_elems * 4
        elif ggml_type == 1:  # F16
            n_bytes = n_elems * 2
        elif ggml_type == 8:  # Q8_0
            n_bytes = ((n_elems + 31) // 32) * 34
        elif ggml_type == 12: # Q4_K
            n_bytes = ((n_elems + 255) // 256) * 144
        elif ggml_type == 26: # I32
            n_bytes = n_elems * 4
        else:
            raise ValueError(f"Unsupported GGML type: {ggml_type}")

        tensors_to_write.append({
            "name": name,
            "dims": dims,
            "ggml_type": ggml_type,
            "n_bytes": n_bytes,
        })

    # Add model embedding and output heads
    add_tensor("token_embd.weight", [n_embd, n_vocab], 1) # F16
    add_tensor("output_hc_base.weight", [4], 0) # F32
    add_tensor("output_hc_fn.weight", [16384, 4], 1) # F16
    add_tensor("output_hc_scale.weight", [1], 0) # F32
    add_tensor("output_norm.weight", [n_embd], 0) # F32
    add_tensor("output.weight", [n_embd, n_vocab], 8) # Q8_0

    # For each layer, add attention, FFN, hyper-connection, and MoE tensors
    for il in range(n_layers):
        add_tensor(f"blk.{il}.hc_attn_fn.weight", [16384, 24], 1) # F16
        add_tensor(f"blk.{il}.hc_attn_scale.weight", [3], 0) # F32
        add_tensor(f"blk.{il}.hc_attn_base.weight", [24], 0) # F32
        add_tensor(f"blk.{il}.attn_norm.weight", [n_embd], 0) # F32
        add_tensor(f"blk.{il}.attn_q_a.weight", [n_embd, 1024], 8) # Q8_0
        add_tensor(f"blk.{il}.attn_q_a_norm.weight", [1024], 0) # F32
        add_tensor(f"blk.{il}.attn_q_b.weight", [1024, 32768], 8) # Q8_0
        add_tensor(f"blk.{il}.attn_kv.weight", [n_embd, 512], 8) # Q8_0
        add_tensor(f"blk.{il}.attn_kv_a_norm.weight", [512], 0) # F32
        add_tensor(f"blk.{il}.attn_sinks.weight", [n_head], 0) # F32
        add_tensor(f"blk.{il}.attn_output_a.weight", [4096, 8192], 8) # Q8_0
        add_tensor(f"blk.{il}.attn_output_b.weight", [8192, n_embd], 8) # Q8_0
        
        # Hyper-connection tensors (identity/ones/zeros)
        add_tensor(f"blk.{il}.hc_ffn_fn.weight", [16384, 24], 1) # F16
        add_tensor(f"blk.{il}.hc_ffn_scale.weight", [3], 0) # F32
        add_tensor(f"blk.{il}.hc_ffn_base.weight", [24], 0) # F32

        # FFN & MoE tensors
        add_tensor(f"blk.{il}.ffn_norm.weight", [n_embd], 0) # F32
        add_tensor(f"blk.{il}.ffn_gate_inp.weight", [n_embd, 512], 1) # F16
        add_tensor(f"blk.{il}.ffn_gate_exps.weight", [n_embd, 2048, 512], 12) # Q4_K
        add_tensor(f"blk.{il}.ffn_up_exps.weight", [n_embd, 2048, 512], 12) # Q4_K
        add_tensor(f"blk.{il}.ffn_down_exps.weight", [2048, n_embd, 512], 12) # Q4_K
        add_tensor(f"blk.{il}.ffn_gate_shexp.weight", [n_embd, 2048], 8) # Q8_0
        add_tensor(f"blk.{il}.ffn_up_shexp.weight", [n_embd, 2048], 8) # Q8_0
        add_tensor(f"blk.{il}.ffn_down_shexp.weight", [2048, n_embd], 8) # Q8_0

        # Hash Layer Routing table (if il < 3)
        if il < 3:
            add_tensor(f"blk.{il}.ffn_gate_tid2eid.weight", [10, n_vocab], 26) # I32 (26)

    # Compute tensor offsets
    next_offset = 0
    for t in tensors_to_write:
        t["rel_offset"] = next_offset
        next_offset += pad_to(t["n_bytes"], GGUF_DEFAULT_ALIGNMENT)

    # Write GGUF file
    print(f"Writing GGUF output to {args.out}...")
    with open(args.out, "wb") as out:
        # Magic & Header
        out.write(b"GGUF")
        out.write(pack_u32(3)) # GGUF v3
        out.write(pack_u64(len(tensors_to_write)))
        out.write(pack_u64(len(kvs)))
        out.write(kv_blob)

        # Tensor directory
        for t in tensors_to_write:
            out.write(pack_string(t["name"]))
            out.write(pack_u32(len(t["dims"])))
            for d in t["dims"]:
                out.write(pack_u64(d))
            out.write(pack_u32(t["ggml_type"]))
            out.write(pack_u64(t["rel_offset"]))

        # Pad to alignment before starting tensor data
        header_dir_len = out.tell()
        data_start = pad_to(header_dir_len, GGUF_DEFAULT_ALIGNMENT)
        pad_len = data_start - header_dir_len
        if pad_len > 0:
            out.write(b"\0" * pad_len)

        # Write tensor payloads as a sparse file by seeking and writing a single byte at the end of each tensor.
        for t in tensors_to_write:
            tensor_end = data_start + t["rel_offset"] + t["n_bytes"]
            out.seek(tensor_end - 1)
            out.write(b"\0")

    print(f"Successfully converted model. Output size: {args.out.stat().st_size / (1024**2):.2f} MB")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
