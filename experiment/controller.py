"""Matched cut selection with optional gain, persistence, and dwell controls."""

from __future__ import annotations

from dataclasses import dataclass


POLICIES = (
    "edge-only", "cloud-only", "fixed-split", "greedy",
    "H", "K", "D", "HK", "HD", "KD", "HKD",
)


@dataclass(frozen=True)
class CutCost:
    cut: int
    edge_ms: float
    server_ms: float
    payload_bytes: int


@dataclass(frozen=True)
class Decision:
    cut: int
    candidate: int
    gain: float
    changed: bool
    supporting_epochs: int
    predicted_ms: dict[int, float]


class Controller:
    def __init__(
        self, costs: list[CutCost], policy: str, *, fixed_cut: int | None = None,
        margin: float = 0.08, persistence: int = 3, dwell: int = 8,
        rtt_ms: float = 0.0,
    ) -> None:
        if policy not in POLICIES:
            raise ValueError(f"Unknown policy {policy}")
        self.costs = {item.cut: item for item in costs}
        self.last_cut = max(self.costs)
        if set(self.costs) != set(range(self.last_cut + 1)):
            raise ValueError("Profile must include every cut from zero to final")
        if policy == "fixed-split" and (fixed_cut is None or not 0 < fixed_cut < self.last_cut):
            raise ValueError("fixed-split needs an interior --fixed-cut chosen on calibration data")
        self.policy = policy
        self.fixed_cut = fixed_cut
        self.margin = margin if "H" in policy else 0.0
        self.persistence = persistence if "K" in policy else 1
        self.dwell = dwell if "D" in policy else 0
        self.rtt_ms = rtt_ms
        self.current = self.last_cut  # safe local initial state
        self.last_switch_epoch = -10**9
        self.candidate: int | None = None
        self.support = 0

    def choose(self, epoch: int, bandwidth_mbps: float) -> Decision:
        if bandwidth_mbps <= 0:
            raise ValueError("Bandwidth must be positive")
        predicted = {}
        for cut, item in self.costs.items():
            transfer_ms = 0.0 if cut == self.last_cut else 8 * item.payload_bytes / (1000 * bandwidth_mbps)
            rtt_ms = 0.0 if cut == self.last_cut else self.rtt_ms
            predicted[cut] = item.edge_ms + item.server_ms + transfer_ms + rtt_ms
        best = min(predicted, key=predicted.get)
        old = self.current
        gain = (predicted[old] - predicted[best]) / max(predicted[old], 1e-9)

        if self.policy in ("edge-only", "cloud-only", "fixed-split"):
            self.current = {"edge-only": self.last_cut, "cloud-only": 0,
                            "fixed-split": self.fixed_cut}[self.policy]
            return Decision(self.current, best, gain, False, 0, predicted)

        supported = best != old and gain >= self.margin
        if supported:
            self.support = self.support + 1 if self.candidate == best else 1
            self.candidate = best
        else:
            self.candidate = None
            self.support = 0
        changed = supported and self.support >= self.persistence and epoch - self.last_switch_epoch >= self.dwell
        if changed:
            self.current = best
            self.last_switch_epoch = epoch
            self.candidate = None
            self.support = 0
        return Decision(self.current, best, gain, changed, self.support, predicted)
