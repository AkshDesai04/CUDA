```cpp
/*
Instead of an `if`, can you use arithmetic to prevent out-of-bounds access (e.g. `C[index] = ...` where index is somehow clamped)? Is this better or worse than a branch?

Thinking:
The core question is whether a branch (the classic `if (idx < N) { ... }`) is preferable to a branchless approach that attempts to clamp the index using arithmetic operations.  
A branch has two major implications:
1. **Correctness** – The branch guarantees that no out‑of‑bounds write occurs because the body is simply not executed.
2. **Performance** – On GPUs, divergent branches within a warp can lead to serialization, but if the branch is simple and the divergent paths are short, the cost is often low.

A branchless clamp might look like:
```
int clamped = min(idx, N-1);
C[clamped] = (idx < N) ? A[idx] + B[idx] : 0.0f;
```
However, this still involves a conditional (the `? :`), and the compiler may or may not eliminate the branch. Even if we try to write:
```
int clamped = min(idx, N-1);
C[clamped] = A[idx] + B[idx];
```
this **does not** prevent out‑of‑bounds access for indices ≥ N, because we still write to memory at `clamped`, which could be N-1 for all out‑of‑bounds indices.  
Thus, purely arithmetic clamping without any form of masking typically does **not** guarantee safety. A proper branchless solution would still require a mask that nullifies the write for out‑of‑bounds indices:
```
int mask = idx < N;            // 1 if valid, 0 otherwise
int clamped = min(idx, N-1);
C[clamped] = mask * (A[idx] + B[idx]);
```
But this uses the comparison to create the mask, which is effectively the same as a branch in terms of correctness.

Performance-wise, for large kernels where most threads hit the same branch (e.g., all indices are valid except a few at the end), the branch divergence cost is negligible and the branch is simpler and safer.  
If the kernel is highly divergent, a branchless approach that eliminates divergence could be faster, but it requires careful handling to avoid memory corruption.  

In summary:
- A branch is safer and often sufficiently fast on GPUs.
- An arithmetic clamp without proper masking can lead to out‑of‑bounds writes.
- A branchless mask approach is possible but still relies on a comparison that may compile to a branch.
- The decision should be based on profiling: measure warp divergence and memory traffic for both variants in the specific application context.
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

// Kernel that uses a branch to guard out-of-bounds writes
__global__ void kernel_branch(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}

// Kernel that attempts to clamp index using arithmetic (with a mask)
__global__ void kernel_clamp(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Create a mask that is 1 for valid indices and 0 otherwise
    int mask = idx < N;                     // 1 if idx < N, else 0
    int clamped = idx < N ? idx : N - 1;    // clamp to last valid index

    // Compute value; for out-of-bounds idx the mask will zero it out
    float val = (float)mask * (A[idx] + B[idx]);

    // Only write if idx is valid; this avoids out-of-bounds writes
    if (mask)
        C[clamped] = val;
}

int main()
{
    const int N = 1 << 20;  // 1M elements
    const int bytes = N * sizeof(float);

    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C_branch = (float*)malloc(bytes);
    float *h_C_clamp = (float*)malloc(bytes);

    // Initialize host arrays
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = sinf(i * 0.01f);
        h_B[i] = cosf(i * 0.01f);
    }

    // Device memory
    float *d_A, *d_B, *d_C_branch, *d_C_clamp;
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C_branch, bytes);
    cudaMalloc((void**)&d_C_clamp, bytes);

    // Copy data to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Launch kernels
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    kernel_branch<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C_branch, N);
    kernel_clamp<<<blocks, threadsPerBlock>>>(d_A, d_B, d_C_clamp, N);

    cudaDeviceSynchronize();

    // Copy results back
    cudaMemcpy(h_C_branch, d_C_branch, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_clamp, d_C_clamp, bytes, cudaMemcpyDeviceToHost);

    // Verify results
    int errors = 0;
    for (int i = 0; i < N; ++i)
    {
        if (fabs(h_C_branch[i] - h_C_clamp[i]) > 1e-5f)
        {
            errors++;
            if (errors <= 10)
            {
                printf("Mismatch at %d: branch=%f, clamp=%f\n", i, h_C_branch[i], h_C_clamp[i]);
            }
        }
    }
    if (errors == 0)
        printf("Results match!\n");
    else
        printf("Found %d mismatches.\n", errors);

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C_branch);
    free(h_C_clamp);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C_branch);
    cudaFree(d_C_clamp);

    return 0;
}
```