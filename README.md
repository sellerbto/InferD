# InferD
Distributed inference framework. 

## Algorithms & techniques adopted from **[InferD — Distributed LLM Inference Engine](https://github.com/sellerbto/InferD/tree/defence)**

- **DHT-based peer discovery (Kademlia-like lookups)**  
  Lightweight metadata (shard → peers, capacity/health hints) is stored in a DHT so clients/coordinators can resolve shard locations without a central index.

- **Capacity-aware consistent hashing for placement**  
  Shards are placed using consistent hashing weighted by node capacity (CPU/GPU, RAM, bandwidth), enabling predictable placement and biasing stronger peers to host larger/more shards.

- **Model partitioning & shard addressing**  
  Support for multiple sharding granularities (layer-wise, contiguous parameter blocks, tensor slices) with stable shard IDs so shards can be located and addressed independently.

- **Pipelined model-parallel execution with micro-batching**  
  Execution graphs are pipelined across shards; micro-batching keeps nodes utilized while keeping per-request latency bounded.

- **Activation streaming & compression**  
  Activations are streamed between peers (not fully materialized centrally). Optional activation compression/quantization and delta-encoding reduce bandwidth and memory peaks.

- **Replica discovery & speculative re-routing**  
  Replication metadata in the DHT enables fast failover. Speculative/backup RPCs and simple replica-selection heuristics mitigate stragglers.

- **Edge caching & reuse**  
  Caching of embeddings, recent activations or partial results at edge peers to accelerate repeated or history-heavy requests (useful for chat and next-edit scenarios).

- **Dynamic rebalancing & network-aware scheduling**  
  Nodes publish capacity/health; placement and scheduling adapt gradually with an emphasis on minimizing end-to-end latency and avoiding large data migrations.

These components implement the core distributed routing, placement, execution and robustness patterns that InferD explores while combining them with capacity-aware scheduling and practical bandwidth/latency optimizations.

To run distributed system, execute ```sh run.sh```

To send inference requests, execute ```uv run send_message.py```
