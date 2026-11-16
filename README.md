# Evaluating Hysteresis-Aware Adaptive DNN Split Inference

This repository contains the research project **“Evaluating Hysteresis-Aware Adaptive DNN Split Inference under Fluctuating Edge-Cloud Network Conditions.”** The project investigates how deep neural network (DNN) inference can be divided between a resource-constrained edge node and a more powerful cloud node while network quality changes over time.

## Research question

Under which patterns and rates of network change can a hysteresis-aware adaptive partitioning policy reduce P95 inference latency and service-level objective (SLO) violations compared with edge-only, cloud-only, fixed-split, and greedy adaptive approaches—without introducing excessive repartitioning overhead?

## Proposed approach

The planned runtime policy selects a DNN split point using a configurable improvement threshold, a required number of consecutive favorable intervals, and a minimum dwell time. These hysteresis controls are intended to prevent frequent split changes caused by brief or noisy network fluctuations while retaining the ability to respond to sustained changes.

The experimental testbed is expected to use:

- Python, PyTorch, and Docker
- Edge and cloud nodes with controlled compute resources
- MobileNetV2 and ResNet18 on CIFAR-10
- HTTP or gRPC-based split inference
- Linux traffic control for repeatable network emulation
- Steady, step-change, bursty, and periodic network scenarios

## Evaluation

The hysteresis-aware policy will be compared with four baselines:

- Edge-only inference
- Cloud-only inference
- A fixed split selected offline
- A greedy adaptive split policy

The evaluation will focus on median, P95, and P99 end-to-end latency; SLO violation rate; throughput; communication volume; switching rate; adaptation delay; controller overhead; and prediction agreement with an unsplit reference model. Repeated runs, matched traces, and confidence intervals are planned to support reproducible comparisons.

## Repository contents

- [Phase 1 research proposal](G31_i220800_i220764_Evaluating_Hysteresis_Aware_Adaptive_DNN_Split_Inference_under_Fluctuating_Edge_Cloud_Network_Conditions.pdf)
- [Current two-column manuscript source](main.tex)
- [Stylized controller simulation and its run-level outputs](simulation/)
- [CIFAR-10 two-node split-inference implementation](experiment/)

The simulation outputs are calculated from declared assumptions and are not DNN latency measurements. The `experiment/` implementation is ready to be configured for a measured study; no benchmark logs or trained checkpoints are committed.

## Researchers

- Zohaib Hassan (22I-0800), Primary Researcher
- Fahad Saleem (22I-0764), Co-Researcher

## Status

The manuscript includes a reproducible preliminary controller simulation. The two-node PyTorch code has not been run for this repository, so its performance remains to be measured. The research gap and expected DNN outcomes remain hypotheses.
