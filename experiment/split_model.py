"""CIFAR-10 stage boundaries shared by training, profiling, edge, and server."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn
from torchvision.models import mobilenet_v2, resnet18


class ResNetHead(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.pool = backbone.avgpool
        self.fc = backbone.fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(torch.flatten(self.pool(x), 1))


class MobileNetHead(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.classifier = backbone.classifier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = nn.functional.adaptive_avg_pool2d(x, (1, 1))
        return self.classifier(torch.flatten(x, 1))


class StageNet(nn.Module):
    def __init__(self, stages: list[nn.Module]):
        super().__init__()
        self.stages = nn.ModuleList(stages)

    @property
    def final_cut(self) -> int:
        return len(self.stages)

    def forward_range(self, x: torch.Tensor, start: int, end: int) -> torch.Tensor:
        if not 0 <= start <= end <= self.final_cut:
            raise ValueError(f"Invalid stage range [{start}, {end})")
        for stage in self.stages[start:end]:
            x = stage(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_range(x, 0, self.final_cut)


def build_model(name: str) -> StageNet:
    if name == "resnet18":
        base = resnet18(weights=None, num_classes=10)
        # Preserve spatial detail at 32 x 32; the same architecture is used on both nodes.
        base.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        base.maxpool = nn.Identity()
        return StageNet([
            nn.Sequential(base.conv1, base.bn1, base.relu, base.maxpool),
            base.layer1, base.layer2, base.layer3, base.layer4, ResNetHead(base),
        ])
    if name == "mobilenet_v2":
        base = mobilenet_v2(weights=None, num_classes=10)
        layers = list(base.features.children())
        return StageNet([
            nn.Sequential(*layers[:4]), nn.Sequential(*layers[4:7]),
            nn.Sequential(*layers[7:14]), nn.Sequential(*layers[14:]),
            MobileNetHead(base),
        ])
    raise ValueError(f"Unsupported model: {name}")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(path: str | Path, device: torch.device) -> tuple[str, StageNet, str]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("num_classes") != 10:
        raise ValueError("Checkpoint must have a CIFAR-10 ten-class head")
    name = payload["model_name"]
    model = build_model(name)
    model.load_state_dict(payload["state_dict"], strict=True)
    model.to(device).eval()
    return name, model, sha256_file(path)
