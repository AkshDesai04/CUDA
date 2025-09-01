**Arithmetic intensity** is a metric that quantifies how many floating‑point (or other) arithmetic operations a program performs per byte of data it moves to/from memory.  
Mathematically,

\[
\text{Arithmetic Intensity} = \frac{\text{Number of arithmetic operations}}{\text{Number of memory operations (bytes or words)}}
\]

A high arithmetic intensity indicates that the computation does a lot of work for each byte it has to read or write, which usually means the program is **compute‑bound** rather than **memory‑bound**.  
A low arithmetic intensity suggests that the program spends most of its time fetching/storing data, making it memory‑bound.

---

### How kernel fusion improves arithmetic intensity

In GPU programming, a **kernel** is a function that runs on the device.  
When multiple kernels are launched sequentially:

1. Each kernel typically loads its input from global memory, does some work, and writes results back.
2. The data written by the first kernel must be written to memory and later read by the next kernel (or written again if a new kernel writes it).

This results in **extra memory traffic**: intermediate results are moved through global memory even though they are only needed for the next kernel’s computation.

**Kernel fusion** merges two or more kernels into a single kernel that performs all the work in one launch:

- **Eliminates intermediate global‑memory writes/reads.** All intermediate values can stay in registers or shared memory, which are orders of magnitude faster and have much lower bandwidth cost.
- **Reduces the total number of memory operations.** Since each piece of data is fetched only once and written only once, the denominator in the arithmetic‑intensity ratio decreases.
- **Increases the number of arithmetic operations per memory operation** because the same data is reused more times within the fused kernel.

As a result, the arithmetic intensity of the fused kernel is higher, leading to better utilization of the GPU’s compute resources and often a measurable performance gain.