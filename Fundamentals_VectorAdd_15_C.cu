**Arithmetic intensity** is a measure of how many arithmetic (flop) operations are performed per memory operation (load or store). It is usually expressed as:

\[
\text{Arithmetic Intensity} = \frac{\text{Number of floating‑point operations}}{\text{Number of bytes transferred from/to memory}}
\]

* **Why it matters**  
  - GPUs (and many modern processors) have a huge disparity between computational throughput and memory bandwidth.  
  - If a kernel has **low arithmetic intensity**, the processor spends most of its time waiting for data to arrive from memory, leading to poor utilization of the arithmetic units.  
  - If a kernel has **high arithmetic intensity**, it can keep the arithmetic units busy while the memory subsystem feeds it data, resulting in better overall performance.

* **Typical example**  
  ```text
  // Simple vector addition: C[i] = A[i] + B[i]
  // 1 addition per element, 3 loads and 1 store = 4 memory operations
  // Arithmetic intensity ≈ 1 flop / 4 memory ops = 0.25
  ```

* **Fusing kernels to improve intensity**  
  - **Kernel fusion** merges two or more kernels into a single kernel, so intermediate results can be kept in registers or shared memory instead of being written to global memory.  
  - This reduces the number of global memory accesses: intermediate results no longer have to be fetched again in the next kernel.  
  - Consequently, the denominator in the intensity ratio shrinks while the numerator (the total number of flops performed in the fused kernel) stays the same or increases.  
  - Example:  
    - Kernel 1: `D[i] = A[i] * B[i]`  
    - Kernel 2: `E[i] = D[i] + C[i]`  
    - **Fused**: `E[i] = A[i] * B[i] + C[i]`  
    - Memory ops reduced from 4 loads + 1 store per element to 3 loads + 1 store, while still performing 2 flops per element. Arithmetic intensity improves from 0.5 to 0.667.

By carefully designing fused kernels and using fast on‑chip memory (registers, shared memory), you can greatly increase arithmetic intensity and thereby extract more performance from the GPU.