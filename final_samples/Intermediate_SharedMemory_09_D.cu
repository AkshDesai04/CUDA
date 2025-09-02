```cuda
/*
Write a kernel where `output[i] = shared_mem[threadIdx.x * 2]`. Profile this. On modern GPUs, this may be handled by the cache, but on older ones it would cause 2-way conflicts.

Thinking:
1. Shared memory in CUDA is divided into banks. On compute capability 2.x and 3.x, there are 32 banks; each bank holds 4 bytes. A bank conflict occurs when two or more threads in a warp access different addresses that map to the same bank.
2. Accessing shared_mem[threadIdx.x * 2] causes two consecutive threads (e.g., thread 0 and thread 128 on a 256-thread warp) to access addresses that are offset by 2 * sizeof(float). Since the bank index is computed as address / 4 bytes modulo number of banks, multiplying by 2 doubles the bank index step. For a 256-thread block, threadIdx.x ranges 0..255; threadIdx.x * 2 ranges 0..510. When we take modulo 256 (the number of shared memory banks in 32 banks * 8? Actually bank count is 32, but each bank holds 4 bytes, so bank index = (offset / 4) % 32. For threadIdx.x * 2 * sizeof(float), the offset in bytes is 4 * threadIdx.x * 2 = 8 * threadIdx.x. Bank index = (8 * threadIdx.x / 4) % 32 = (2 * threadIdx.x) % 32. So each bank index is used by two threads: threadIdx.x and threadIdx.x + 16, producing 2‑way conflicts on older GPUs.
3. On newer GPUs (compute capability 6.0+), the shared memory controller can handle these conflicts using caching or reordering, reducing the penalty. However, to demonstrate the potential penalty, we write the kernel and run it on a system with an older GPU (or emulate the conflict by forcing a shared memory size that is too small).
4. The kernel loads data from global memory into shared memory, synchronizes, and then each thread reads from shared_mem[threadIdx.x * 2] and writes to output. The shared memory is allocated with extern __shared__ float shared_mem[] and we launch with a size of blockDim.x * sizeof(float).
5. For profiling, we use cudaEvent_t to measure kernel execution time. We also query device properties to print compute capability, which indicates whether the GPU might handle the conflict efficiently.
6. The host code allocates input/output arrays, initializes input, copies to device, launches kernel, copies back, prints a few elements, and frees memory.
7. This example demonstrates how shared memory access patterns can influence performance and how to profile them in CUDA.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Kernel that demonstrates 2-way shared memory bank conflicts
__global__ void conflictKernel(float *output, const float *input, int N)
{
    extern __shared__ float shared_mem[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    if (idx < N) {
        shared_mem[threadIdx.x] = input[idx];
    }

    __syncthreads();

    // Each thread reads from shared_mem[threadIdx.x * 2 % blockDim.x]
    if (idx < N) {
        int bankIndex = (threadIdx.x * 2) % blockDim.x;
        output[idx] = shared_mem[bankIndex];
    }
}

int main()
{
    // Problem size
    const int N = 256;
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    // Allocate host memory
    float *h_input  = (float*)malloc(N * sizeof(float));
    float *h_output = (float*)malloc(N * sizeof(float));

    // Initialize input
    for (int i = 0; i < N; ++i) {
        h_input[i] = static_cast<float>(i);
    }

    // Allocate device memory
    float *d_input  = nullptr;
    float *d_output = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_input,  N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_output, N * sizeof(float)));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    // Query device properties
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("Running on device: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);

    // Timing events
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Launch kernel
    size_t sharedMemSize = blockSize * sizeof(float);
    CHECK_CUDA(cudaEventRecord(start));
    conflictKernel<<<gridSize, blockSize, sharedMemSize>>>(d_output, d_input, N);
    CHECK_CUDA(cudaEventRecord(stop));

    // Wait for completion
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Compute elapsed time
    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("Kernel execution time: %f ms\n", ms);

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify a few elements
    printf("First 10 output values:\n");
    for (int i = 0; i < 10; ++i) {
        printf("output[%d] = %f\n", i, h_output[i]);
    }

    // Clean up
    free(h_input);
    free(h_output);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
```