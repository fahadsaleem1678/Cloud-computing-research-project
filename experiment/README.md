# Two-node split-inference study

This directory contains the CIFAR-10 model, cut controller, profiling tools, HTTP suffix service, shaped-link runner, and log analysis for the study described in `../main.tex`. It is an implementation package. No measured result is claimed until the commands are run and their raw logs are retained.

## Environment

Use Python 3.11 or later on Linux for link shaping. Install a matching PyTorch and Torchvision pair using the [official PyTorch installer](https://pytorch.org/get-started/locally/), then install `requirements.txt`. The runner needs `tc` from `iproute2` and permission to control a **dedicated test interface**. It replaces that interface's root queueing discipline during a run and removes it afterward. Use a network namespace or disposable veth; do not point it at a production interface.

```bash
python -m pip install -r requirements.txt
python train.py --model resnet18 --output checkpoints/resnet18.pt
python train.py --model mobilenet_v2 --output checkpoints/mobilenet_v2.pt
python verify_splits.py --checkpoint checkpoints/resnet18.pt
```

Training uses 45,000 CIFAR-10 training images and a fixed 5,000-image validation split. The test set is not used to select a checkpoint. The trained checkpoint is needed on both nodes. Record package versions and the checkpoint SHA-256 in the final study.

## Profile the same checkpoint on both nodes

Run the first command on the edge host and the second on the server host. Copy the resulting JSON files to the edge host. The profiler times every prefix and suffix, but the runner uses only edge prefix and server suffix measurements. Repeat profiling after any hardware, precision, or checkpoint change.

```bash
python profile.py --checkpoint checkpoints/resnet18.pt --output profiles/edge.json --device cpu
python profile.py --checkpoint checkpoints/resnet18.pt --output profiles/server.json --device cpu
```

The two example commands look identical because they are issued on different machines. In a single-machine smoke check, both files can be created locally, but that does not establish a two-node performance result.

## Start the server and run a policy

On the server, bind to the test-network address:

```bash
python server.py --checkpoint checkpoints/resnet18.pt --host 0.0.0.0 --port 8765
```

On the edge, run one policy and one trace per invocation. Supply the server's reachable test-network address, plus the name of a dedicated Linux interface. Choose the fixed cut using calibration runs and keep it fixed for all held-out traces. Repeat with matched seeds and image order for `edge-only`, `cloud-only`, `fixed-split`, `greedy`, `H`, `K`, `D`, `HK`, `HD`, `KD`, and `HKD`. Rotate policy execution order across replications to reduce drift.

```bash
python run.py \
  --checkpoint checkpoints/resnet18.pt \
  --edge-profile profiles/edge.json --server-profile profiles/server.json \
  --server-url http://SERVER_TEST_IP:8765 \
  --trace traces/step.csv --policy HKD --seed 1729 \
  --tc-interface veth-edge --output runs/resnet18-step-HKD-1729.jsonl
```

The runner writes one JSON record per request and a companion `.metadata.json` file. The trace CSV supplies **configured** bandwidth, delay, and loss to both `tc` and the controller. It is not an independent measurement of achieved link quality. Collect link probes separately and include them in the final analysis. The `--no-shaping` option exists for development only; its logs must not be treated as a network-trace comparison. This runner processes one request at a time, so throughput under concurrent offered load needs a separate extension.

## Analyze logs

```bash
python analyze.py runs/*.jsonl \
  --runs-output runs/run_summary.csv \
  --paired-output runs/paired_effects.csv
```

The analysis includes failures in the deadline-violation denominator, reports completed-request latency percentiles, counts switches, checks prediction agreement, and forms paired HKD-minus-greedy bootstrap intervals for runs with the same model, checkpoint, trace, and seed. Keep the raw JSONL files, checkpoint hashes, trace configuration, hardware specification, and a copy of the environment alongside any result table.

## Scope

The `simulation/` directory at repository root holds the separate stylized controller study cited in the article. Its component times and network conditions are assumptions. The code here is for collecting measured CIFAR-10 split-inference results, which have not yet been supplied.
