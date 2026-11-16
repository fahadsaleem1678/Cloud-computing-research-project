"""Train and save a CIFAR-10 checkpoint without using the test set for selection."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from split_model import build_model, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["resnet18", "mobilenet_v2"], required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    augmented = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    plain = transforms.Compose([transforms.ToTensor(), normalize])
    train_data = datasets.CIFAR10(args.data_root, train=True, download=True, transform=augmented)
    val_data = datasets.CIFAR10(args.data_root, train=True, download=False, transform=plain)
    indices = torch.randperm(len(train_data), generator=torch.Generator().manual_seed(args.seed)).tolist()
    train_loader = DataLoader(Subset(train_data, indices[:45000]), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Subset(val_data, indices[45000:]), batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = build_model(args.model).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()
    best = -1.0
    history = []
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(images), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(images)
        model.eval()
        correct = 0
        with torch.inference_mode():
            for images, targets in val_loader:
                logits = model(images.to(device)).cpu()
                correct += (logits.argmax(1) == targets).sum().item()
        accuracy = correct / len(val_loader.dataset)
        history.append({"epoch": epoch + 1, "train_loss": total_loss / len(train_loader.dataset), "val_accuracy": accuracy})
        print(json.dumps(history[-1]))
        if accuracy > best:
            best = accuracy
            torch.save({"model_name": args.model, "num_classes": 10,
                        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                        "seed": args.seed, "epoch": epoch + 1, "val_accuracy": accuracy}, args.output)
        scheduler.step()
    args.output.with_suffix(".json").write_text(json.dumps({"history": history, "best_val_accuracy": best,
                                                             "checkpoint_sha256": sha256_file(args.output)}, indent=2))


if __name__ == "__main__":
    main()
