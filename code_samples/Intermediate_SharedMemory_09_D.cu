```cpp
/* 
Write a kernel where `output[i] = shared_mem[threadIdx.x * 2]`. Profile this. On modern GPUs, this may be handled by the cache, but on older ones it would cause 2‑way conflicts.

Thought process:
1. The goal is to expose a classic 2‑way shared‑memory bank conflict on GPUs that do not automatically hide it with a cache.  
2. To generate a conflict we load data into `shared_mem[threadIdx.x]` for each thread in the block, then later each thread reads from `shared_mem[threadIdx.x * 2]`.  
3. On GPUs with 32 banks, indices that are multiples of 2 will map to banks 0,2,4,…,30. For the first 32 threads each bank is used once, but for threads 32‑63 the same banks are used again → 2‑way conflict.  
4. We keep the shared memory size equal to the block size (`blockDim.x`) so that `shared_mem[threadIdx.x * 2]` is out of bounds for the last 64 threads; we guard the read so the kernel does not fault but still illustrates the conflict.  
5. The host code allocates input data, copies it to the device, launches the kernel with `blockDim.x = 128` and times the kernel using CUDA events.  
6. After the kernel finishes we copy back the result and print the first few values to verify correctness.  
7. The timing measurement will let the user compare performance on different GPU architectures.  
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Kernel that creates a 2‑way shared memory conflict
__global__ void conflictKernel(const int* __restrict__ d_in, int* __restrict__ d_out, size_t N)
{
    // Allocate shared memory of size equal to blockDim.x
    extern __shared__ int s_mem[];

    // Global thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory
    if (idx < N) {
        s_mem[threadIdx.x] = d_in[idx];
    }

    // Synchronize to ensure all loads are complete
    __syncthreads();

    // Read from shared memory with potential conflict
    if (idx < N) {
        size_t srcIdx = static_cast<size_t>(threadIdx.x) * 2;
        // Guard against out‑of‑bounds accesses
        if (srcIdx < blockDim.x) {
            d_out[idx] = s_mem[srcIdx];
        } else {
            // For threads that would read out of bounds, write a sentinel
            d_out[idx] = -1;
        }
    }
}

int main()
{
    // Problem size
    const size_t N = 1 << 20; // 1M elements
    const size_t bytes = N * sizeof(int);

    // Allocate host memory
    int* h_in  = (int*)malloc(bytes);
    int* h_out = (int*)malloc(bytes);

    // Initialize input with some pattern
    for (size_t i = 0; i < N; ++i) {
        h_in[i] = static_cast<int>(i);
    }

    // Allocate device memory
    int *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    // Copy input to device
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    // Kernel launch configuration
    const int threadsPerBlock = 128;
    const int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;
    size_t sharedMemBytes = threadsPerBlock * sizeof(int);

    // Timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warm up run
    conflictKernel<<<blocksPerGrid, threadsPerBlock, sharedMemBytes>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    // Start timing
    cudaEventRecord(start, 0);

    // Launch kernel
    conflictKernel<<<blocksPerGrid, threadsPerBlock, sharedMemBytes>>>(d_in, d_out, N);

    // Stop timing
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Compute elapsed time
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("Kernel execution time: %f ms\n", ms);

    // Copy result back to host
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    // Verify a few results
    printf("Sample output values:\n");
    for (int i = 0; i < 10; ++i) {
        printf("h_out[%d] = %d\n", i, h_out[i]);
    }

    // Clean up
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
```