Thread divergence occurs when the threads of a warp take different execution paths due to a conditional statement that evaluates to true for some threads and false for others. In CUDA, a warp is the unit of execution on a GPU, consisting of 32 threads that are scheduled and executed in lock‑step.  

When a warp encounters a branching instruction (e.g., an `if` statement), the hardware must resolve the branch. Since all threads in the warp share the same program counter, the GPU cannot execute two different branches simultaneously for the same warp. Instead, it serially executes the code for each distinct path taken by the threads in that warp:

1. **Mask creation** – The GPU creates two masks: one for the threads that take the true branch and one for those that take the false branch.
2. **Branch execution** – The warp executes the true branch for the subset of threads whose mask is active, while the other threads are idle. After the true branch completes, the warp then executes the false branch for the remaining threads.
3. **Re‑merging** – Once both branches have finished, the warp continues together with all 32 threads again.

Because the two paths are executed sequentially rather than concurrently, thread divergence can significantly reduce performance. The worst case is when only one thread in a warp follows one branch and the remaining 31 follow the other; the warp will effectively waste 31/32 of its execution resources while one thread is idle.  

To minimize thread divergence:

- Align data so that neighboring threads execute similar paths.
- Use predication (e.g., `__syncthreads()` or `if` statements that apply to entire warps) rather than per‑thread branches when possible.
- Structure algorithms so that control flow is uniform across threads in the same warp.

In summary, thread divergence happens when not all threads in a warp follow the same execution path. The GPU resolves this by serially executing each branch, which can lead to underutilization of the warp’s resources and decreased throughput.