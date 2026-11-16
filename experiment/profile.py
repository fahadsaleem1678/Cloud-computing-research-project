"""Profile every prefix and suffix on one host; run once per edge/server host."""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import torch

from split_model import load_checkpoint


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def time_call(fn, device: torch.device, repeats: int) -> float:
    with torch.inference_mode():
        for _ in range(5):
            fn()
        synchronize(device)
        samples = []
        for _ in range(repeats):
            start = time.perf_counter_ns()
            fn()
            synchronize(device)
            samples.append((time.perf_counter_ns() - start) / 1e6)
    return sorted(samples)[len(samples) // 2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    device = torch.device(args.device)
    name, model, checkpoint_hash = load_checkpoint(args.checkpoint, device)
    image = torch.rand(1, 3, 32, 32, device=device)
    rows = []
    for cut in range(model.final_cut + 1):
        with torch.inference_mode():
            activation = model.forward_range(image, 0, cut)
        edge_ms = time_call(lambda: model.forward_range(image, 0, cut), device, args.repeats)
        server_ms = time_call(lambda: model.forward_range(activation, cut, model.final_cut), device, args.repeats)
        buffer = io.BytesIO()
        torch.save(activation.detach().cpu(), buffer)
        rows.append({"cut": cut, "prefix_ms": edge_ms, "suffix_ms": server_ms,
                     "payload_bytes": len(buffer.getvalue()) if cut < model.final_cut else 0})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"model_name": name, "checkpoint_sha256": checkpoint_hash,
                                      "device": str(device), "torch_version": torch.__version__,
                                      "cuts": rows}, indent=2))
    print(args.output)


if __name__ == "__main__":
    main()
