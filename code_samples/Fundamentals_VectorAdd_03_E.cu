The boundary check is typically **not** computationally expensive.  
Here’s why:

1. **Cheap Arithmetic**  
   The check usually consists of a few simple integer comparisons (e.g., `if (idx < N)` or `if (i < rows && j < cols)`). These operations cost only a few clock cycles on a GPU core, far less than the cost of a full memory load/store or a complex mathematical operation.

2. **Branch Divergence Is Often Minimal**  
   In a CUDA kernel, threads are executed in groups called warps (32 threads). If the boundary condition is true for all threads in a warp, the branch is taken uniformly and there is no divergence: the warp proceeds without any penalty. Divergence only occurs when some threads in the warp satisfy the condition while others do not. In many typical kernels, especially when the problem size is a multiple of the warp size or when the grid is padded to avoid edge cases, almost all warps stay fully inside the valid region, so no divergence occurs.

3. **Branch Prediction & Hardware Optimisation**  
   Modern GPUs are designed to handle simple branches efficiently. The hardware predicts the outcome of the comparison and can pre‑speculate the execution path, making the overhead negligible.

4. **Memory‑Bound Kernels Dominate**  
   For many GPU kernels, the primary performance bottleneck is memory bandwidth, not instruction execution. The cost of a few comparison instructions pales in comparison to the latency of global memory accesses.

**When It Might Be Expensive**  
- If the kernel has many warps that straddle the boundary (e.g., irregular data shapes or small grids), divergence can become significant, leading to serialized execution of the divergent branches.  
- In tight loops where the boundary check is executed millions of times, even a cheap branch could add up, but this is rarely the limiting factor compared to memory traffic or compute‑heavy arithmetic.

In summary, a boundary check is a very cheap branch and, for full warps where divergence is absent, it usually does not affect performance. The primary factors that limit performance are typically memory access patterns, arithmetic intensity, and other GPU resource constraints.