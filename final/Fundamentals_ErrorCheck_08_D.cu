```cuda
/* 
Aim: Write a kernel that has a 1-in-a-million chance of accessing out of bounds. Run it in a loop. 
Without error checking, it might run correctly 99% of the time, highlighting the danger of silent errors.

Thinking:
- To create a 1-in-a-million probability for each thread, we generate a simple pseudo‑random
  value based on the thread's global index and apply a modulus test.
- The kernel writes to a global array of size N. Most of the time it writes to the element
  at its own index, but when the random test triggers (with probability 1/1e6) it writes
  to element N, which is out of bounds.
- The host launches this kernel many times in a loop. Since we do not perform
  cudaGetLastError() or cudaDeviceSynchronize() after each launch, any out‑of‑bounds
  write may go unnoticed, and the program will often finish successfully.
- This demonstrates how silent errors can arise when error checking is omitted.
*/

#include <cuda_runtime.h>
#include <stdio.h>

#define N 1024                      // Size of the array
#define THREADS_PER_BLOCK 256       // Threads per block
#define NUM_ITERATIONS 1000000      // Number of kernel launches

// Kernel that occasionally writes out of bounds
__global__ void riskyKernel(int *arr, int size)
{
    // Compute global thread index
    unsigned int gid = blockIdx.x * blockDim.x + threadIdx.x;

    // Simple LCG-based pseudo‑random number derived from gid
    unsigned int rnd = gid;
    rnd = (rnd * 1103515245u + 12345u) & 0x7fffffffu; // 31‑bit LCG

    // 1‑in‑1,000,000 chance to access out of bounds
    if (rnd % 1000000 == 0)
    {
        // Intentionally write beyond the array
        arr[size] = gid;  // Out‑of‑bounds write
    }
    else
    {
        // Normal write within bounds
        if (gid < size)
            arr[gid] = gid;
    }
}

int main(void)
{
    int *d_arr;
    size_t bytes = N * sizeof(int);

    // Allocate device memory
    cudaMalloc((void**)&d_arr, bytes);

    // Determine grid dimensions
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Run the kernel many times without error checking
    for (int i = 0; i < NUM_ITERATIONS; ++i)
    {
        riskyKernel<<<blocks, THREADS_PER_BLOCK>>>(d_arr, N);
        // Intentionally omitted: cudaGetLastError(), cudaDeviceSynchronize()
    }

    // Cleanup (optional, but good practice)
    cudaFree(d_arr);
    cudaDeviceReset();

    return 0;
}
```