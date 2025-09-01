/*
Use a grid-stride loop to initialize a very large array to a specific value.

Thinking:
- The core requirement is to use a grid‑stride loop in a CUDA kernel so that each thread
  can work on multiple elements if the array is larger than the total number of threads.
- We choose a simple kernel that takes a pointer to device memory, the number of elements
  (N), and the value to write.  Each thread starts at its global index and then jumps
  forward by the total grid stride (blockDim.x * gridDim.x) until it reaches or
  surpasses N.
- For a very large array we allocate device memory using cudaMalloc.  The size is
  specified as a 64‑bit integer (size_t or long long) so we can go up to several
  billions of elements without overflow.
- Host code sets up timing with cudaEvent to measure the launch time, launches the
  kernel with a reasonable block size (e.g., 256), and calculates the grid size
  based on the number of elements.
- We include a small error‑checking macro to simplify error handling for CUDA API calls.
- After the kernel completes we optionally copy back a few elements to verify the
  value.  Printing a few values is enough for sanity checking without overwhelming
  output.
- The code is fully self‑contained in a single .cu file and can be compiled with nvcc.
*/

#include <cstdio>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                   \
    do {                                                                    \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\n",           \
                    __FILE__, __LINE__, err, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                            \
        }                                                                   \
    } while (0)

// Kernel that uses a grid‑stride loop to initialize an array to a given value
__global__ void initArrayKernel(float *arr, size_t N, float val) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (; idx < N; idx += stride) {
        arr[idx] = val;
    }
}

int main(void) {
    // Size of the array: 1 << 28 = 268,435,456 elements (~1 GB of float)
    const size_t N = (1ULL << 28);
    const float initVal = 3.14159f;

    // Allocate device memory
    float *d_arr = nullptr;
    CHECK_CUDA(cudaMalloc(&d_arr, N * sizeof(float)));

    // Timing events
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Configure kernel launch parameters
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    // Record start event
    CHECK_CUDA(cudaEventRecord(start, 0));

    // Launch kernel
    initArrayKernel<<<gridSize, blockSize>>>(d_arr, N, initVal);
    CHECK_CUDA(cudaGetLastError());

    // Record stop event and synchronize
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Calculate elapsed time
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));

    printf("Initialized %zu elements to %.6f in %.3f ms\n", N, initVal, ms);

    // Optional: Verify a few elements
    const int verifyCount = 5;
    float h_verify[verifyCount];
    CHECK_CUDA(cudaMemcpy(h_verify, d_arr, verifyCount * sizeof(float), cudaMemcpyDeviceToHost));
    printf("Sample values after initialization:\n");
    for (int i = 0; i < verifyCount; ++i) {
        printf("  [%d] = %.6f\n", i, h_verify[i]);
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_arr));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
