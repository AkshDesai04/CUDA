```cpp
/*
Aim of the program: Explain what memory coalescing is. (Threads in a warp accessing contiguous memory locations in a single transaction).

Thinking process:
1. Memory coalescing is an optimization in CUDA that reduces the number of memory transactions between device memory and the GPU's memory hierarchy.
2. In CUDA, threads are grouped into warps (32 threads). If all threads in a warp access consecutive memory addresses that fall into the same memory segment, the GPU can fetch that segment with a single memory transaction, rather than 32 separate transactions.
3. This reduces memory bandwidth usage and improves performance.
4. To demonstrate this, we will write a simple CUDA kernel that performs a coalesced read from an array: each thread reads the element at index `i` (where `i` is its global thread index). Since each thread accesses consecutive elements, the accesses are coalesced.
5. For comparison, we also provide a non-coalesced kernel where each thread accesses an element that is offset by a large stride, leading to many separate memory transactions.
6. The program will allocate an array, initialize it, launch both kernels, and copy back the results. We also time the execution of each kernel to give a crude sense of the performance difference.
7. The program uses standard CUDA runtime API calls. No external libraries are needed. It compiles with `nvcc` and runs on any GPU supporting compute capability 3.0+.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <chrono>

#define CHECK_CUDA(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),     \
                    cudaGetErrorString(err));                           \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

const int ARRAY_SIZE = 1 << 20; // 1M elements
const int BLOCK_SIZE = 256;

// Coalesced kernel: each thread reads array[idx]
__global__ void coalesced_kernel(const int *in, int *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] + 1; // simple operation
    }
}

// Non-coalesced kernel: each thread reads array[idx * stride]
__global__ void non_coalesced_kernel(const int *in, int *out, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int src_idx = idx * stride;
    if (src_idx < n) {
        out[idx] = in[src_idx] + 1;
    }
}

int main() {
    // Allocate host memory
    int *h_in = (int*)malloc(ARRAY_SIZE * sizeof(int));
    int *h_out = (int*)malloc(ARRAY_SIZE * sizeof(int));
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        h_in[i] = i;
    }

    // Allocate device memory
    int *d_in, *d_out;
    CHECK_CUDA(cudaMalloc((void**)&d_in, ARRAY_SIZE * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void**)&d_out, ARRAY_SIZE * sizeof(int)));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_in, h_in, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice));

    // Determine grid size
    int num_blocks = (ARRAY_SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Launch coalesced kernel and time it
    cudaEvent_t startCoalesced, stopCoalesced;
    CHECK_CUDA(cudaEventCreate(&startCoalesced));
    CHECK_CUDA(cudaEventCreate(&stopCoalesced));
    CHECK_CUDA(cudaEventRecord(startCoalesced));
    coalesced_kernel<<<num_blocks, BLOCK_SIZE>>>(d_in, d_out, ARRAY_SIZE);
    CHECK_CUDA(cudaEventRecord(stopCoalesced));
    CHECK_CUDA(cudaEventSynchronize(stopCoalesced));
    float msCoalesced;
    CHECK_CUDA(cudaEventElapsedTime(&msCoalesced, startCoalesced, stopCoalesced));

    // Copy result back (optional)
    CHECK_CUDA(cudaMemcpy(h_out, d_out, ARRAY_SIZE * sizeof(int), cudaMemcpyDeviceToHost));

    // Launch non-coalesced kernel with stride 32 and time it
    int stride = 32;
    int outSize = (ARRAY_SIZE + stride - 1) / stride; // number of output elements
    int num_blocks_non = (outSize + BLOCK_SIZE - 1) / BLOCK_SIZE;

    cudaEvent_t startNonCoalesced, stopNonCoalesced;
    CHECK_CUDA(cudaEventCreate(&startNonCoalesced));
    CHECK_CUDA(cudaEventCreate(&stopNonCoalesced));
    CHECK_CUDA(cudaEventRecord(startNonCoalesced));
    non_coalesced_kernel<<<num_blocks_non, BLOCK_SIZE>>>(d_in, d_out, ARRAY_SIZE, stride);
    CHECK_CUDA(cudaEventRecord(stopNonCoalesced));
    CHECK_CUDA(cudaEventSynchronize(stopNonCoalesced));
    float msNonCoalesced;
    CHECK_CUDA(cudaEventElapsedTime(&msNonCoalesced, startNonCoalesced, stopNonCoalesced));

    // Print timings
    printf("Coalesced kernel time: %f ms\n", msCoalesced);
    printf("Non-coalesced kernel time: %f ms\n", msNonCoalesced);

    // Clean up
    free(h_in);
    free(h_out);
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));

    CHECK_CUDA(cudaEventDestroy(startCoalesced));
    CHECK_CUDA(cudaEventDestroy(stopCoalesced));
    CHECK_CUDA(cudaEventDestroy(startNonCoalesced));
    CHECK_CUDA(cudaEventDestroy(stopNonCoalesced));

    return 0;
}
```