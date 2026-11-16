"""Check every stage boundary against the unsplit model before benchmarking."""

from __future__ import annotations

import argparse
import torch

from split_model import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()
    device = torch.device(args.device)
    _, model, digest = load_checkpoint(args.checkpoint, device)
    torch.manual_seed(4721)
    images = torch.rand(8, 3, 32, 32, device=device)
    with torch.inference_mode():
        reference = model(images)
        for cut in range(model.final_cut + 1):
            first = model.forward_range(images, 0, cut)
            result = model.forward_range(first, cut, model.final_cut)
            error = (result - reference).abs().max().item()
            if not torch.allclose(result, reference, rtol=1e-5, atol=args.atol):
                raise SystemExit(f"Cut {cut} failed: max absolute difference {error}")
            print(f"cut={cut} max_abs_diff={error:.9g}")
    print(f"All cuts matched; checkpoint_sha256={digest}")


if __name__ == "__main__":
    main()
