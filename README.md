# Prefill–Decode Disaggregation Prototype

**Benchmarking split vs. monolithic inference for large language models**

---

## Overview

This repository contains a **prototype implementation and benchmark** of *prefill–decode disaggregation* — a systems-level technique used in modern LLM serving (e.g., **DistServe**) to improve throughput and reduce latency.

In traditional (monolithic) inference, both the **prefill** (encoding the input prompt) and **decode** (token generation) stages are executed sequentially on a single GPU stream.  
Based on contemporary research, this prototype demonstrates how splitting these stages into **independent workers** allows workload overlap, improving GPU utilization and inference time, while reducing TPOT under concurrent workloads.

---

## Key Features

- **Two Execution Modes**
  - **Monolithic Mode:** Standard `model.generate()` pipeline.
  - **Split Mode (Prefill–Decode):** Disaggregated worker setup using multiprocessing queues to emulate distributed inference.

- **Consistent Latency Metrics**
  - **TTFT (Time to First Token):** uniformly measured as  
    `prefill_end - enqueue_time` (DistServe-style) for both modes.
  - **TPOT (Time Per Output Token):** `(end_time - first_token_time) / (token_count - 1)`
  - Ensures *apples-to-apples* comparison between monolithic and split runs.

- **10 Prompt Workloads**
  - Covers all **light/heavy prefill–decode combinations**, designed to stress distinct pipeline phases:
    | Prefill | Decode | Example Type | Purpose |
    |----------|---------|---------------|----------|
    | Light | Light | Q&A / lookup | latency baseline |
    | Light | Heavy | essay / story | decode throughput |
    | Heavy | Light | long context / short answer | prefill overlap |
    | Heavy | Heavy | document generation | full pipeline stress |

- **Unified Metric Comparison**
  - Compares TTFT, TPOT and decode tokens per second

---


## Requirements

- Python ≥ 3.10  
- PyTorch ≥ 2.1  
- Transformers ≥ 4.40  
- CUDA GPU

Install dependencies:
```bash
pip install transformers accelerate bitsandbytes
```
