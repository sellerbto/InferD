# InferD

Distributed inference framework inspired by [Petals](https://arxiv.org/abs/2209.01188) and [SWARM Parallelism](https://arxiv.org/abs/2301.11913).

## Overview

InferD implements pipeline parallelism for transformer models across decentralized network. Each node hosts subset of model layers (stages) and collaborates via Kademlia DHT. DHT maps stage numbers to available nodes `stage_id -> {node_id: {load, capacity}}`. New nodes connect via bootstrap nodes and announce their stage. No global state sync, nodes query DHT on-demand.

Client sends request to stage-0 node with input text. Node tokenizes, runs forward pass through local layers, queries DHT for next stage, selects least-loaded peer. Hidden states serialized as base64-encoded tensors and forwarded via HTTP. Process repeats until final stage returns decoded token. Stateless HTTP RPC between stages. Each node holds 1/N of model weights. Forward pass requires N-1 network hops per token.

Nodes periodically rebalance to equalize stage load. If current stage has low load and another stage is overloaded, node migrates by reloading model weights for new stage. TaskScheduler tracks active tasks, Balancer compares load distribution via DHT.

System is decentralized, no single point of failure, DHT replicated across all nodes. PathFinder retries on node unavailability and triggers rebalance. Failed nodes removed from DHT via timeout. Multiple nodes can serve same stage for redundancy. Limitations: no checkpointing for in-flight requests on crash, DHT convergence delay after topology changes, cold start penalty for stage reassignment.

Model partitioning done in `split_model.py`. FirstStage has embedding + rotary + layers[0:k], StageInner has rotary + layers[k:m], LastStage has rotary + layers[m:end] + norm + lm_head. PathFinder selects next-hop using greedy load-based heuristic. D* Lite prepared but unused. Tensors serialized to JSON with base64 encoding.

## Growth Areas

Activate D* Lite for latency-aware routing, request checkpointing at stage boundaries, batching multiple requests at same stage, KV-cache sharing between requests, persistent connections (gRPC/WebSocket), hierarchical DHT for large swarms, distributed tracing and monitoring.

## Quick Start

```bash
python split_model.py
INITIAL_STAGE=0 NODE_NAME=node0 python petals/run_node.py
INITIAL_STAGE=1 NODE_NAME=node1 BOOTSTRAP_NODES=node0:7050 python petals/run_node.py
```
