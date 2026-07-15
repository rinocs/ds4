#!/usr/bin/env python3
"""Download Qwen3/Qwen2.5 weights and configuration from Hugging Face."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path

def download_file(url: str, dest: Path, token: str | None = None) -> None:
    print(f"Downloading {dest.name}...")
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
        
    try:
        with urllib.request.urlopen(req) as response:
            meta = response.info()
            file_size = int(meta.get("Content-Length", 0))
            
            with open(dest, "wb") as f:
                block_size = 8192
                downloaded = 0
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    f.write(buffer)
                    downloaded += len(buffer)
                    if file_size:
                        percent = downloaded * 100 / file_size
                        sys.stdout.write(f"\rProgress: {percent:.2f}% ({downloaded}/{file_size} bytes)")
                        sys.stdout.flush()
                sys.stdout.write("\n")
    except Exception as e:
        print(f"\nerror downloading {url}: {e}", file=sys.stderr)
        raise

def main() -> int:
    parser = argparse.ArgumentParser(description="Download Qwen3/Qwen2.5 Coder weights from Hugging Face.")
    parser.add_argument("--repo", default="Qwen/Qwen2.5-Coder-7B-Instruct", help="Hugging Face repository ID")
    parser.add_argument("--out-dir", default="qwen3_model", type=Path, help="Directory to save downloaded weights")
    parser.add_argument("--token", help="Hugging Face authorization token")
    args = parser.parse_args()

    # Read HF token from cache if not explicitly provided
    token = args.token
    if not token:
        hf_token_cache = Path.home() / ".cache" / "huggingface" / "token"
        if hf_token_cache.exists():
            token = hf_token_cache.read_text().strip()

    print(f"Querying repository metadata for {args.repo}...")
    api_url = f"https://huggingface.co/api/models/{args.repo}"
    req = urllib.request.Request(api_url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")

    try:
        with urllib.request.urlopen(req) as response:
            repo_info = json.loads(response.read().decode())
    except Exception as e:
        print(f"error querying Hugging Face API: {e}", file=sys.stderr)
        return 1

    # Extract config.json and weight files (.safetensors or .bin)
    files_to_download: list[str] = []
    siblings = repo_info.get("siblings", [])
    
    # We want config.json first
    has_config = False
    for s in siblings:
        rpath = s.get("rpath")
        if rpath == "config.json":
            has_config = True
        elif rpath.endswith(".safetensors") or (rpath.endswith(".bin") and "pytorch_model" in rpath):
            files_to_download.append(rpath)
            
    if not has_config:
        print("warning: config.json not found in repository listings", file=sys.stderr)
    else:
        files_to_download.insert(0, "config.json")

    if not files_to_download:
        print("error: no matching weight or config files found in the repository", file=sys.stderr)
        return 1

    print(f"Found {len(files_to_download)} files to download.")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for file in files_to_download:
        url = f"https://huggingface.co/{args.repo}/resolve/main/{file}"
        dest = args.out_dir / file
        
        # Create subdirectories if files are nested
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            download_file(url, dest, token)
        except Exception:
            return 1

    print(f"\nSuccessfully downloaded model weights and config to {args.out_dir}")
    print(f"You can now run convert_qwen3.py to build the DwarfStar GGUF model:")
    print(f"  python3 gguf-tools/convert_qwen3.py --model-dir {args.out_dir} --out qwen3.gguf")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
