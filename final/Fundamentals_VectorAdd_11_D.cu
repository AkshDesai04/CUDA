/*
Is there a performance difference between a standard launch (one thread per element) and a grid-stride launch (many elements per thread) for a problem that fits in one launch? Time both.

The idea behind this benchmark is to compare two common CUDA programming patterns:

1. **Standard launch** – every element is processed by a single thread.  
   The kernel simply computes an output index from block and thread IDs,
   checks bounds, and performs the operation.  
   For a problem that fits in one launch this is the most straightforward
   approach.  It has the benefit of having minimal loop overhead,
   no branching, and each thread doing exactly one iteration.

2. **Grid‑stride launch** – a single kernel is launched with a smaller
   number of threads.  Each thread processes many elements by looping
   over the grid with a fixed stride equal to the total number of
   threads.  
   The kernel code becomes:

   ```cuda
   for (int i = globalIdx; i < N; i += stride) { … }
   ```

   This pattern is useful when the workload may exceed the number of
   threads you can launch in a single grid or when you want to reuse
   the same kernel for different problem sizes without changing the
   launch configuration.  However, it introduces loop overhead and
   may incur a small amount of branch divergence when the final loop
   iteration goes out of bounds.

For a problem that *does* fit in one launch (i.e., the total number
of elements is less than the maximum number of threads that can be
launched in a single grid) we expect the standard launch to be slightly
faster because each thread does just one memory load/store and the
kernel avoids the loop machinery.  The grid‑stride version will still
work but may have a tiny overhead from the loop control.

This benchmark will:

- Allocate large vectors `A` and `B`, and compute `C = A + B`.
- Run the computation twice: once with a standard launch and once
  with a grid‑stride launch.
- Time both executions using CUDA events.
- Print the elapsed times in milliseconds.

The code below is self‑contained and can be compiled with `nvcc`:
```
nvcc -O2 benchmark.cu -o benchmark
```
Then run `./benchmark`.  The program reports the two execution times
and a rough speedup factor.

*/

#include <stdio.h>
#include <cuda_runtime.h>

// Macro for checking CUDA errors following a kernel launch or API call
#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",    \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),          \
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Kernel that processes one element per thread
__global__ void kernelStandard(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        C[idx] = A[idx] + B[idx];
}

// Kernel that processes many elements per thread using a grid‑stride loop
__global__ void kernelGridStride(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; idx < N; idx += stride)
        C[idx] = A[idx] + B[idx];
}

int main()
{
    const int N = 1 << 24;               // 16 million elements (~64 MB per array)
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *hA = (float*)malloc(size);
    float *hB = (float*)malloc(size);
    float *hC = (float*)malloc(size);
    if (!hA || !hB || !hC) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input data
    for (int i = 0; i < N; ++i) {
        hA[i] = (float)i;
        hB[i] = (float)(N - i);
    }

    // Allocate device memory
    float *dA, *dB, *dC;
    CHECK_CUDA(cudaMalloc((void**)&dA, size));
    CHECK_CUDA(cudaMalloc((void**)&dB, size));
    CHECK_CUDA(cudaMalloc((void**)&dC, size));

    // Copy inputs to device
    CHECK_CUDA(cudaMemcpy(dA, hA, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice));

    // Determine launch configuration
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // --------------------
    // Standard launch
    // --------------------
    CHECK_CUDA(cudaEventRecord(start));
    kernelStandard<<<blocksPerGrid, threadsPerBlock>>>(dA, dB, dC, N);
    CHECK_CUDA(cudaEventRecord(stop));

    // Wait for kernel to finish
    CHECK_CUDA(cudaEventSynchronize(stop));
    float elapsedStandard;
    CHECK_CUDA(cudaEventElapsedTime(&elapsedStandard, start, stop));

    // Copy result back to host (optional)
    CHECK_CUDA(cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost));

    // --------------------
    // Grid‑stride launch
    // --------------------
    CHECK_CUDA(cudaEventRecord(start));
    kernelGridStride<<<blocksPerGrid, threadsPerBlock>>>(dA, dB, dC, N);
    CHECK_CUDA(cudaEventRecord(stop));

    CHECK_CUDA(cudaEventSynchronize(stop));
    float elapsedGridStride;
    CHECK_CUDA(cudaEventElapsedTime(&elapsedGridStride, start, stop));

    // Copy result back to host (optional)
    CHECK_CUDA(cudaMemcpy(hC, dC, size, cudaMemcpyDeviceToHost));

    // --------------------
    // Output results
    // --------------------
    printf("Standard launch time   : %6.2f ms\n", elapsedStandard);
    printf("Grid‑stride launch time: %6.2f ms\n", elapsedGridStride);
    printf("Speedup (Standard / Grid‑stride): %.2fx\n",
           elapsedGridStride / elapsedStandard);

    // Clean up
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(dA));
    CHECK_CUDA(cudaFree(dB));
    CHECK_CUDA(cudaFree(dC));
    free(hA);
    free(hB);
    free(hC);

    return 0;
}
