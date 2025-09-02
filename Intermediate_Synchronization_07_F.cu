```cuda
/*
Aim of the program:
"What is a better approach than a global lock for managing contention? (Lock-free algorithms, privatization, finer-grained locks)."

Thinking:
In CUDA there is no concept of a traditional global mutex or lock that can be used across many threads in parallel.  Instead we have atomic operations that provide lock‑free synchronization for a single memory location.  For more complex data structures or when many threads need to update different parts of a structure, we can use privatization (each thread block or even each thread owns a private copy of the data) and then combine the results, or we can use finer‑grained locks by giving each element its own lock or using atomicCAS on that element.  This program demonstrates three approaches to sum an array of integers:
  1. A naïve global atomicAdd on a single global counter (lock‑free, but all threads contend on the same location).
  2. Privatization: each thread block accumulates a local sum in shared memory and then atomically adds the block sum to a global counter (finer‑grained contention).
  3. A purely lock‑free algorithm that uses atomicCAS to increment a shared counter without a global lock, and shows how the atomicAdd is already lock‑free.

The kernels illustrate that while atomicAdd is lock‑free, contention on a single memory location can still hurt performance.  By moving the aggregation into thread‑block local memory we reduce contention and obtain a better approach than a global lock.

The code below can be compiled with `nvcc` and executed on a CUDA‑capable GPU.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <assert.h>

#define ARRAY_SIZE (1 << 20)      // 1M elements
#define BLOCK_SIZE 256

// Kernel 1: Naïve global atomicAdd (lock‑free but high contention)
__global__ void sum_global_atomic(const int *d_in, int *d_sum)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < ARRAY_SIZE)
    {
        atomicAdd(d_sum, d_in[idx]);  // lock‑free atomic addition
    }
}

// Kernel 2: Privatization – each block accumulates in shared memory then atomicAdd
__global__ void sum_block_private(const int *d_in, int *d_sum)
{
    __shared__ int localSum;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadIdx.x == 0) localSum = 0;
    __syncthreads();

    int val = 0;
    if (idx < ARRAY_SIZE)
        val = d_in[idx];

    atomicAdd(&localSum, val);   // atomicAdd within the block (low contention)
    __syncthreads();

    if (threadIdx.x == 0)
        atomicAdd(d_sum, localSum);  // one atomicAdd per block
}

// Kernel 3: Pure lock‑free increment using atomicCAS (demonstrates atomic operations)
__global__ void sum_atomic_cas(const int *d_in, int *d_sum)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= ARRAY_SIZE) return;

    int val = d_in[idx];
    // Increment the global sum using atomicCAS in a loop
    int old, newVal;
    do {
        old = atomicAdd(d_sum, 0);          // read current value
        newVal = old + val;                 // compute new value
    } while (atomicCAS(d_sum, old, newVal) != old); // try to write new value
}

int main()
{
    // Allocate host memory
    int *h_in = (int*)malloc(ARRAY_SIZE * sizeof(int));
    assert(h_in != NULL);

    // Initialize input array with ones for easy sum verification
    for (int i = 0; i < ARRAY_SIZE; ++i)
        h_in[i] = 1;

    // Allocate device memory
    int *d_in;
    int *d_sum;
    cudaMalloc((void**)&d_in, ARRAY_SIZE * sizeof(int));
    cudaMalloc((void**)&d_sum, sizeof(int));

    // Copy data to device
    cudaMemcpy(d_in, h_in, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel 1
    cudaMemset(d_sum, 0, sizeof(int));
    int grid = (ARRAY_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sum_global_atomic<<<grid, BLOCK_SIZE>>>(d_in, d_sum);
    cudaDeviceSynchronize();
    int h_sum1;
    cudaMemcpy(&h_sum1, d_sum, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Sum with global atomicAdd: %d\n", h_sum1);

    // Launch kernel 2
    cudaMemset(d_sum, 0, sizeof(int));
    sum_block_private<<<grid, BLOCK_SIZE>>>(d_in, d_sum);
    cudaDeviceSynchronize();
    int h_sum2;
    cudaMemcpy(&h_sum2, d_sum, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Sum with block privatization: %d\n", h_sum2);

    // Launch kernel 3
    cudaMemset(d_sum, 0, sizeof(int));
    sum_atomic_cas<<<grid, BLOCK_SIZE>>>(d_in, d_sum);
    cudaDeviceSynchronize();
    int h_sum3;
    cudaMemcpy(&h_sum3, d_sum, sizeof(int), cudaMemcpyDeviceToHost);
    printf("Sum with atomicCAS loop: %d\n", h_sum3);

    // Verify correctness
    int expected = ARRAY_SIZE * 1;
    if (h_sum1 == expected && h_sum2 == expected && h_sum3 == expected)
        printf("All sums correct.\n");
    else
        printf("Discrepancy in sums!\n");

    // Clean up
    cudaFree(d_in);
    cudaFree(d_sum);
    free(h_in);
    return 0;
}
```