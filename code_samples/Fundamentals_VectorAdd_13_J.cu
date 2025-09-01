**Warp divergence** occurs when threads within the same warp (typically 32 threads on NVIDIA GPUs) take different execution paths because of branching statements such as `if`, `for`, or the ternary operator. Since a warp executes in lock‑step, all threads must complete the same instruction before any can proceed. When divergence happens, the warp is split into *sub‑warps* that execute each branch sequentially. The threads that are not executing a particular branch are idle (they are effectively “masked out”) during that segment.

---

### Example: `(A[i] > B[i]) ? X : Y`

- **If all threads in the warp satisfy `A[i] > B[i]`**, every thread executes `X`.  
  The warp runs the branch once, and performance is as if no branching existed.

- **If some threads satisfy the condition and others don’t**, the warp diverges:
  1. The warp executes `X` for the threads that take that path, while the others are masked.
  2. The warp then executes `Y` for the remaining threads, while the former `X` threads are masked.

  In the worst case (half the threads take `X`, half take `Y`), the warp spends roughly twice the time it would have taken if all threads had taken a single path. In general, the total execution time is proportional to the *maximum* number of active threads among all divergent branches.

---

### Performance impact

- **Serialization**: Divergence turns parallel execution into a serial sequence of the different paths, reducing throughput.
- **Resource under‑utilization**: While one subset of threads is executing a branch, the others are idle, so the GPU’s computational resources (ALUs, memory bandwidth) are not fully utilized.
- **Latency**: The time per warp increases, which can dominate the kernel’s runtime if divergence is frequent or the branches involve expensive operations.

In practice, divergent branches can reduce a kernel’s efficiency by up to a factor of two or more, depending on how many threads take each path and how the branches are structured. To mitigate this, programmers often:

- Re‑structure code to make all threads in a warp follow the same path.
- Use predication or data‑parallel techniques that avoid conditional branches.
- Align data and loops so that warp execution remains uniform.

---

**Bottom line:** In `(A[i] > B[i]) ? X : Y`, if some threads in a warp take `X` and others take `Y`, the warp will execute both branches sequentially, causing the warp to run roughly twice as long as it would if all threads took the same branch, thereby degrading performance.