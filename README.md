# CUDA Performance Fundamentals

A set of CUDA microbenchmarks exploring how thread-block configuration, global memory access patterns, and explicit shared memory reuse affect GPU kernel performance.

The goal of these experiments is to build intuition for how CUDA execution and the GPU memory hierarchy translate into measured performance.

## Experiments

### 1. Threads per Block

A vector add kernel was used to study how launch configuration affects execution time while keeping the workload fixed.

**Workload:** 10,000,000 elements

| Threads / Block | Blocks | Median Time (ms) |
|---:|---:|---:|
| 32  | 312,500 | 0.247808 |
| 64  | 156,250 | 0.126976 |
| 128 | 78,125  | 0.095232 |
| 256 | 39,063  | 0.094208 |
| 512 | 19,532  | 0.095232 |

Performance improved substantially between 32 and 128 threads per block, then effectively plateaued. The lowest measured median time occurred at 256 threads per block, but the difference between 128, 256, and 512 threads was negligible.

This experiment illustrates that increasing the number of threads per block does not indefinitely improve performance. Once the launch configuration exposes sufficient parallelism, other factors become more important than simply adding threads.

---

### 2. Coalesced vs. Strided Global Memory Access

Two copy kernels performed the same amount of work but used different thread-to-memory mappings.

**Matrix size:** 8192 × 8192

| Access Pattern | Median Time (ms) |
|---|---:|
| Coalesced | 0.4352 |
| Strided | 2.58202 |

**Slowdown:** 5.93×  

The coalesced version maps neighboring threads in a warp to nearby memory locations. The strided version changes this mapping so that threads within a warp access memory less efficiently.

Changing only the memory access pattern produced a roughly 5.9× slowdown, demonstrating that GPU performance depends strongly on how threads collectively access global memory, even when the amount of computation is unchanged.

---

### 3. Global Memory vs. Shared Memory Reuse

This experiment studies when explicitly staging data in shared memory becomes worthwhile.

The global memory version repeatedly accesses the source value, while the shared memory version first stages data in shared memory and then reuses it from there. The reuse count is varied while the rest of the workload remains fixed.

**Workload:** 2^24 elements  
**Threads per block:** 256

| Reuse | Global (ms) | Shared (ms) | Global / Shared |
|---:|---:|---:|---:|
| 1  | 0.110080 | 0.111616 | 0.986× |
| 2  | 0.118784 | 0.116736 | 1.018× |
| 4  | 0.129024 | 0.128000 | 1.008× |
| 8  | 0.194560 | 0.168960 | 1.152× |
| 16 | 0.337408 | 0.288256 | 1.171× |
| 32 | 0.628736 | 0.525312 | 1.197× |


At low reuse, shared memory provides little or no benefit because staging data introduces additional work and synchronization. As reuse increases, that cost is amortized and the shared-memory version becomes increasingly faster, reaching approximately a 1.20× speedup at reuse 32.

The broader lesson is that explicit data reuse is not automatically beneficial: the performance gain from reuse must be large enough to outweigh the overhead of staging and synchronization.

## Key Takeaways

- Launch configuration matters, but increasing threads per block eventually reaches a point of diminishing returns.
- Coalesced global memory access can have a large performance advantage over unfavorable thread to memory mappings.
- Shared memory is most useful when data reuse is high enough to amortize the cost of staging and synchronization.
- GPU optimization requires measuring trade-offs rather than assuming that a commonly recommended technique will always improve performance.

## Methodology

Each experiment uses CUDA event timing with warm-up runs before measurement. Multiple timed runs are collected and the median execution time is reported. Outputs are checked for correctness before interpreting performance results.

## Environment

These experiments were run using CUDA on an NVIDIA GPU in Google Colab.

- GPU: NVIDIA A100-SXM4-40GB
- CUDA Compiler: NVIDIA `nvcc` 12.8 (V12.8.93)

## Repository Structure

```text
.
├── vector_add/
│   └── ...
├── coalescing/
│   └── ...
├── shared_memory/
│   └── ...
└── README.md
```

The individual experiment directories contain the CUDA source code and benchmark logic used to produce the results above.
