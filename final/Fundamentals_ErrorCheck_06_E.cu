```cuda
/* 
Aim: Demonstrate its use by checking a kernel that fails due to an out-of-bounds access.

Thinking:
- The goal is to write a small CUDA program that intentionally causes a kernel
  to perform an out‑of‑bounds memory access, so that we can observe and
  handle the resulting error. 
- We'll allocate a device array of a fixed size (e.g. 16 integers).
- The kernel will be launched with 16 threads in a single block, but each
  thread will write to an index that is 10 positions beyond the allocated
  array (e.g. index = tid + 10).  This guarantees an out‑of‑bounds
  write for all threads.
- After launching the kernel we will immediately call
  cudaGetLastError() to fetch the error code set by the kernel launch.
  Then we will call cudaDeviceSynchronize() to make sure the kernel has
  finished executing, because the runtime error may not be detected
  until synchronization.
- If an error is detected we will print the error string via
  cudaGetErrorString().
- Finally we will free the allocated memory and exit.
- The program demonstrates how to catch and report errors arising from
  memory violations in CUDA kernels.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that intentionally writes out of bounds
__global__ void outOfBoundsKernel(int *d_array, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Intentionally write 10 positions beyond the valid range
    int out_of_bounds_idx = idx + 10;
    d_array[out_of_bounds_idx] = idx;  // This will cause an out-of-bounds write
}

int main(void)
{
    const int N = 16;                 // Size of the array
    const size_t bytes = N * sizeof(int);

    int *d_array = NULL;
    cudaError_t err = cudaMalloc((void**)&d_array, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Launch kernel with one block of N threads
    outOfBoundsKernel<<<1, N>>>(d_array, N);

    // Check for launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
    } else {
        // Even if launch succeeded, the kernel may have produced an error during execution.
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel execution error: %s\n", cudaGetErrorString(err));
        } else {
            printf("Kernel executed successfully (unexpected).\n");
        }
    }

    cudaFree(d_array);
    return 0;
}
```