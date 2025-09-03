**Warp Divergence**

In CUDA, a *warp* is a group of 32 threads that execute the same instruction simultaneously on the streaming multiprocessor (SIMT) model.  
When the threads in a warp follow different control‑flow paths—i.e., some take one branch of an `if/else` (or a conditional expression) while others take another—the warp is said to diverge.  

**How divergence is handled**

The hardware resolves divergence by splitting the warp into *sub‑warps* that execute each branch separately:

1. The warp executes the first branch.  
   Only the threads that satisfy the branch condition are *active*; the others are *masked* (they do nothing).
2. Once the first branch finishes, the warp executes the second branch, again masking the inactive threads.

Thus the two branches are executed **serially**, not in parallel.  

**Performance impact**

- The total execution time of the warp becomes the sum of the times for each branch.  
- If the two branches are roughly equal in cost, the warp may take about **twice** as long as it would have if all threads followed the same path.  
- In the worst case (e.g., many branches or highly unbalanced work), the penalty can approach the number of distinct paths (up to 32× for 32‑thread warps if every thread diverges into a unique path).  
- While diverging threads are masked, the CUDA scheduler can keep the GPU busy with other warps, but the overall throughput for that warp is reduced.

**Specific example**

```c
(A[i] > B[i]) ? X : Y
```

If in a given warp some threads satisfy `A[i] > B[i]` and go to `X`, while the rest go to `Y`:

- The warp executes `X` for the active subset, then executes `Y` for the remaining subset.  
- The warp’s execution time is roughly the sum of the two branch times, so performance can degrade by up to a factor of two relative to a divergence‑free kernel.  

This serialization, coupled with idle (masked) cycles, is the principal cost of warp divergence in CUDA.