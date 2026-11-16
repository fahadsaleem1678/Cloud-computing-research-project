"""Run one matched CIFAR-10 policy/trace workload against the suffix server."""

from __future__ import annotations

import argparse
import csv
import io
import json
import platform
import subprocess
import time
from pathlib import Path
from urllib.request import Request, urlopen

import torch
from torchvision import datasets, transforms

from controller import Controller, CutCost, POLICIES
from split_model import load_checkpoint


def read_profile(path: Path) -> dict:
    return json.loads(path.read_text())


def read_trace(path: Path) -> list[dict]:
    with path.open(newline="") as stream:
        rows = [dict(row) for row in csv.DictReader(stream)]
    if not rows or float(rows[0]["second"]) != 0:
        raise ValueError("Trace must start at second zero")
    events = []
    for row in rows:
        events.append({"second": float(row["second"]), "bw_mbps": float(row["bw_mbps"]),
                       "delay_ms": float(row["delay_ms"]), "loss_pct": float(row["loss_pct"])})
    if [r["second"] for r in events] != sorted(set(r["second"] for r in events)):
        raise ValueError("Trace times must be unique and ascending")
    if any(r["bw_mbps"] <= 0 or r["delay_ms"] < 0 or not 0 <= r["loss_pct"] <= 100 for r in events):
        raise ValueError("Trace has invalid link values")
    return events


def trace_state(events: list[dict], elapsed: float) -> dict:
    state = events[0]
    for row in events[1:]:
        if row["second"] > elapsed:
            break
        state = row
    return state


def set_link(interface: str, state: dict) -> None:
    # Use only a dedicated veth or test interface. This replaces its root qdisc.
    subprocess.run(["tc", "qdisc", "replace", "dev", interface, "root", "netem",
                    "rate", f'{state["bw_mbps"]}mbit', "delay", f'{state["delay_ms"]}ms',
                    "loss", f'{state["loss_pct"]}%'], check=True)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def infer(model, image: torch.Tensor, cut: int, server_url: str, device: torch.device, timeout: float) -> tuple[torch.Tensor, int, int]:
    with torch.inference_mode():
        activation = model.forward_range(image.to(device), 0, cut)
        synchronize(device)
        if cut == model.final_cut:
            return activation.cpu(), 0, 0
        buffer = io.BytesIO()
        torch.save(activation.cpu(), buffer)
        body = buffer.getvalue()
        request = Request(server_url.rstrip("/") + "/infer", data=body,
                          headers={"Content-Type": "application/octet-stream", "X-Cut": str(cut)},
                          method="POST")
        with urlopen(request, timeout=timeout) as response:
            result = response.read()
        logits = torch.load(io.BytesIO(result), map_location="cpu", weights_only=True)
        if not isinstance(logits, torch.Tensor):
            raise ValueError("Server response was not a tensor")
        return logits, len(body), len(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--edge-profile", type=Path, required=True)
    parser.add_argument("--server-profile", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--fixed-cut", type=int)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--tc-interface", help="Dedicated Linux test interface; root qdisc will be replaced")
    parser.add_argument("--no-shaping", action="store_true", help="Development only; not a valid network-trace run")
    parser.add_argument("--requests", type=int, default=300)
    parser.add_argument("--rate-hz", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--deadline-ms", type=float, default=120.0)
    parser.add_argument("--timeout-s", type=float, default=15.0)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--margin", type=float, default=0.08)
    parser.add_argument("--persistence", type=int, default=3)
    parser.add_argument("--dwell", type=int, default=8)
    args = parser.parse_args()
    if not args.tc_interface and not args.no_shaping:
        parser.error("Provide --tc-interface for a shaped run, or --no-shaping for development")
    if args.requests < 1 or args.rate_hz <= 0 or args.persistence < 1 or args.dwell < 0 or args.margin < 0:
        parser.error("Invalid request rate or controller parameter")

    device = torch.device(args.device)
    name, model, digest = load_checkpoint(args.checkpoint, device)
    edge_profile, server_profile = read_profile(args.edge_profile), read_profile(args.server_profile)
    for profile in (edge_profile, server_profile):
        if profile["model_name"] != name or profile["checkpoint_sha256"] != digest:
            raise ValueError("Profile model/checkpoint does not match the runner")
    with urlopen(args.server_url.rstrip("/") + "/health", timeout=args.timeout_s) as response:
        health = json.load(response)
    if health["model_name"] != name or health["checkpoint_sha256"] != digest or health["final_cut"] != model.final_cut:
        raise ValueError("Server model/checkpoint differs from the edge")
    e = {row["cut"]: row for row in edge_profile["cuts"]}
    s = {row["cut"]: row for row in server_profile["cuts"]}
    costs = [CutCost(cut, e[cut]["prefix_ms"], 0 if cut == model.final_cut else s[cut]["suffix_ms"],
                     e[cut]["payload_bytes"]) for cut in range(model.final_cut + 1)]
    events = read_trace(args.trace)
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    test_data = datasets.CIFAR10(args.data_root, train=False, download=False, transform=transform)
    indices = torch.randperm(len(test_data), generator=torch.Generator().manual_seed(args.seed)).tolist()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {"checkpoint_sha256": digest, "model_name": name, "trace": str(args.trace),
                "policy": args.policy, "fixed_cut": args.fixed_cut, "seed": args.seed,
                "requests": args.requests, "rate_hz": args.rate_hz, "deadline_ms": args.deadline_ms,
                "device": str(device), "server_device": health["device"],
                "edge_profile": str(args.edge_profile), "server_profile": str(args.server_profile),
                "shaping_enabled": bool(args.tc_interface), "tc_interface": args.tc_interface,
                "torch_version": torch.__version__, "python_version": platform.python_version()}
    args.output.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2))
    controller = Controller(costs, args.policy, fixed_cut=args.fixed_cut, margin=args.margin,
                            persistence=args.persistence, dwell=args.dwell)
    # Warm both processes without scoring these requests.
    first = test_data[indices[0]][0].unsqueeze(0)
    for _ in range(args.warmup):
        infer(model, first, 0, args.server_url, device, args.timeout_s)
        with torch.inference_mode():
            model(first.to(device))
    active = None
    start = time.monotonic()
    try:
        with args.output.open("w", encoding="utf-8") as out:
            for epoch in range(args.requests):
                due = start + epoch / args.rate_hz
                time.sleep(max(0, due - time.monotonic()))
                schedule_lag_ms = max(0.0, (time.monotonic() - due) * 1000)
                elapsed = time.monotonic() - start
                state = trace_state(events, elapsed)
                if state is not active:
                    if args.tc_interface:
                        set_link(args.tc_interface, state)
                    active = state
                controller.rtt_ms = state["delay_ms"]
                image, target = test_data[indices[epoch % len(indices)]]
                image = image.unsqueeze(0)
                created = time.perf_counter_ns()
                decision = controller.choose(epoch, state["bw_mbps"])
                decision_ms = (time.perf_counter_ns() - created) / 1e6
                record = {"epoch": epoch, "request_id": f'{args.seed}-{epoch}', "policy": args.policy,
                          "trace_second": elapsed, "schedule_lag_ms": schedule_lag_ms,
                          "configured_bw_mbps": state["bw_mbps"],
                          "configured_delay_ms": state["delay_ms"], "configured_loss_pct": state["loss_pct"],
                          "cut": decision.cut, "candidate": decision.candidate, "gain": decision.gain,
                          "changed": decision.changed, "decision_ms": decision_ms,
                          "supporting_epochs": decision.supporting_epochs,
                          "predicted_ms": decision.predicted_ms, "target": target}
                try:
                    logits, sent, received = infer(model, image, decision.cut, args.server_url, device, args.timeout_s)
                    finished = time.perf_counter_ns()
                    latency_ms = (finished - created) / 1e6
                    with torch.inference_mode():
                        reference = model(image.to(device)).cpu()
                    record.update({"status": "ok", "latency_ms": latency_ms,
                                   "deadline_violation": latency_ms > args.deadline_ms,
                                   "sent_bytes": sent, "received_bytes": received,
                                   "prediction": logits.argmax(1).item(),
                                   "reference_prediction": reference.argmax(1).item(),
                                   "max_abs_diff": (logits - reference).abs().max().item()})
                except (OSError, ValueError, RuntimeError) as error:
                    record.update({"status": "error", "error": str(error), "latency_ms": None,
                                   "deadline_violation": True, "sent_bytes": None, "received_bytes": None})
                out.write(json.dumps(record) + "\n")
                out.flush()
    finally:
        if args.tc_interface:
            subprocess.run(["tc", "qdisc", "del", "dev", args.tc_interface, "root"], check=False)
    print(args.output)


if __name__ == "__main__":
    main()
