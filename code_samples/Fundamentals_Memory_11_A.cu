/*
 * Constant memory properties in CUDA:
 *
 * 1. Cached:  Constant memory resides in a dedicated on-chip cache (the constant cache). 
 *    When a kernel reads from constant memory, the values are fetched into this cache 
 *    and reused by all threads in the block, reducing memory traffic to the device memory.
 *
 * 2. Read-only for kernels:  Once a kernel is launched, the constant memory is immutable 
 *    for that kernel execution. The kernel can only read from constant memory; attempts 
 *    to write to it will result in a compilation or runtime error.
 *
 * 3. Low latency when all threads in a warp access the same address:  If every thread 
 *    in a warp performs a read from the same constant memory address, the constant cache 
 *    can service the request with a single memory transaction, yielding very low latency 
 *    and maximum throughput. Divergent accesses (different threads reading different 
 *    addresses) will incur multiple transactions and increased latency.
 */