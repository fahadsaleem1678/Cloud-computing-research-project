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
- Experimental code, network profiles, configurations, logs, and analysis scripts will be added as the project progresses.

## Researchers

- Zohaib Hassan (22I-0800), Primary Researcher
- Fahad Saleem (22I-0764), Co-Researcher

## Status

The project is currently in the research proposal and experimental planning phase. The research gap and expected outcomes described here are hypotheses to be tested, not final claims.
