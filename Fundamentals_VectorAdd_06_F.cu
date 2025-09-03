/*
Handle large vectors with boundary checks.

The goal of this CUDA program is to demonstrate how to safely process very large vectors on a GPU by using proper boundary checks within the kernel. When dealing with vectors that may contain more elements than the number of available threads or blocks, it is essential to guard each memory access so that no thread attempts to read or write outside the allocated memory range. 

The implementation follows these key ideas:

1. **Large Vector Size**  
   We define a vector size `N` that can be as large as 2^30 (around one billion elements). This size can easily exceed the limits of a single kernel launch if we naïvely map one thread per element. To accommodate this, we use a **grid‑stride loop** in the kernel, which allows a moderate number of threads to process the entire array in steps.

2. **Boundary Checks**  
   Inside the grid‑stride loop, we check that the current global index `idx` is less than `n` before performing any read or write. This simple condition protects against out‑of‑bounds accesses.

3. **Memory Allocation**  
   Host memory is allocated with `malloc` and initialized with random data. Device memory is allocated with `cudaMalloc`. Because the vectors can be huge, we use `size_t` (64‑bit) for all indices and sizes.

4. **Kernel Launch Configuration**  
   We choose a reasonable block size (e.g., 256 threads) and compute the grid size as `(n + blockSize - 1) / blockSize`. The kernel can launch with as many blocks as the grid size, but the grid‑stride loop ensures that even if we launch fewer blocks, all elements will still be processed.

5. **Verification**  
   After copying the result back to the host, we perform a simple sanity check on a few random elements to ensure the computation was correct.

6. **Error Checking**  
   A helper macro `CUDA_CHECK` is used throughout to catch and report CUDA errors immediately.

By incorporating these strategies, the program can robustly handle vectors that are too large to be processed with a one‑to‑one thread mapping while keeping memory accesses within bounds.