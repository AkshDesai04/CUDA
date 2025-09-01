```cuda
/*
Aim of the program: Run `cuda-memcheck` with the `--tool synccheck` option to look for synchronization errors.

Thinking:
- The user wants to observe how `cuda-memcheck` reports synchronization problems.
- We need to provide a CUDA program that contains a realistic synchronization bug that can be detected by the `synccheck` tool.
- A common scenario is missing a `__syncthreads()` call between threads that write to shared memory and threads that read from it.
- We'll write a simple parallel reduction kernel that intentionally omits a `__syncthreads()` after a write stage.
- The host code will allocate a small array, launch the kernel, copy back the result, and print it.
- When this program is run with `cuda-memcheck --tool synccheck`, it should report a synchronization error due to the missing barrier.
- The code is kept minimal for clarity while still demonstrating the bug.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that performs a simple parallel reduction to sum an array.
// Intentionally missing a __syncthreads() after writing to shared memory.
__global__ void reduce_sum(int *input, int *output, int n)
{
    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // Load elements into shared memory, each thread loads two elements
    int mySum = 0;
    if (i < n)           mySum += input[i];
    if (i + blockDim.x < n) mySum += input[i + blockDim.x];
    sdata[tid] = mySum;
    // *** Missing __syncthreads() here ***  
    // The next step assumes all threads have written their partial sums

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        // *** Missing __syncthreads() after each addition ***
        // This omission will trigger a synchronization error in synccheck
    }

    // Write result for this block to global memory
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

int main()
{
    const int N = 1 << 20; // 1M elements
    const int THREADS_PER_BLOCK = 256;

    int *h_in = (int*)malloc(N * sizeof(int));
    for (int i = 0; i < N; ++i) h_in[i] = 1; // simple data: all ones

    int *d_in, *d_out;
    cudaMalloc((void**)&d_in, N * sizeof(int));

    // Compute number of blocks needed (each block processes 2*THREADS_PER_BLOCK elements)
    int blocks = (N + THREADS_PER_BLOCK * 2 - 1) / (THREADS_PER_BLOCK * 2);
    cudaMalloc((void**)&d_out, blocks * sizeof(int));

    cudaMemcpy(d_in, h_in, N * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel with shared memory size equal to THREADS_PER_BLOCK * sizeof(int)
    reduce_sum<<<blocks, THREADS_PER_BLOCK, THREADS_PER_BLOCK * sizeof(int)>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    // Reduce block results on host
    int *h_out = (int*)malloc(blocks * sizeof(int));
    cudaMemcpy(h_out, d_out, blocks * sizeof(int), cudaMemcpyDeviceToHost);

    long long sum = 0;
    for (int i = 0; i < blocks; ++i) sum += h_out[i];
    printf("Sum = %lld\n", sum);

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    return 0;
}
```