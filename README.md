# InferD
Distributed inference framework. There are key features:

1) Model partitioning & placement  
Split a large model into addressable shards (by layers, contiguous parameter chunks, or tensor slices). Each shard receives a stable ID and a small metadata record (size, memory/compute footprint). Placement uses capacity-aware consistent hashing: peers publish capacity (CPU/GPU, RAM, bandwidth) and the placement policy assigns larger or more shards to stronger peers while keeping shard locality predictable. The goal is to make shards small enough for flexible placement but large enough to avoid excessive RPC overhead. Placement metadata is lightweight so the system can rebalance gradually without moving heavy weights unless necessary.

2) DHT-based discovery & smart routing  
A DHT holds only metadata (shard ID → hosting peer list, capacity/health hints, replica timestamps). Clients and coordinators perform DHT lookups to resolve “where is shard X?” and receive ordered candidates (by latency, load, replica freshness). Because the DHT is decentralized and compact, lookups remain scalable and resilient as peers churn. Routing logic uses the DHT results plus simple heuristics (network proximity, current load) to pick the best peer for each shard call, avoiding a single coordinator bottleneck.

3) Execution planning, pipelining & streaming  
After resolving shard locations, the client or a lightweight planner constructs a distributed execution graph (ordered shard calls + data dependencies). Execution is run as pipelined micro-batches: while shard N works on micro-batch k, shard N+1 processes k-1, etc., to keep all peers busy. Activations are streamed between peers instead of materializing full tensors centrally — this reduces peak memory usage and enables very long models on modest nodes. To reduce latency and bandwidth, the system supports activation compression, quantization, and delta-encoding for incremental edits or “next-edit” style workloads.

4) Robustness, caching & operational concerns  
Shards are replicated (configurable replication factor); the DHT lists replicas so failures and slow nodes can be routed around. Speculative backup RPCs or re-routing to fresh replicas mitigate stragglers. Caching (embeddings, recent activations, partial answers) at edge peers accelerates repeated or history-heavy requests; cache TTLs and consistency rules are tuned per workload. Operationally, monitor per-shard latency, network throughput and cache hit rates; tune shard size, replication factor, micro-batch size and caching policies to balance latency, throughput and resource usage.

To run distributed system, execute ```sh run.sh```

To send inference requests, execute ```uv run send_message.py```
