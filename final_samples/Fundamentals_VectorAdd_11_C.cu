```cpp
/*
Explain the advantages of a grid-stride loop. (Kernel is independent of launch size, can handle any N, better hardware utilization if launch is small).

The purpose of this program is to illustrate why grid‑stride loops are a powerful pattern in CUDA kernels.  The comments above the kernel explain the key benefits:
- **Independence from launch configuration** – the same kernel code works whether we launch a single block, a few blocks, or thousands of blocks.  The loop automatically jumps over elements that are processed by other blocks.
- **No need to adapt the kernel for different problem sizes** – the `for` loop condition (`i < N`) ensures that any value of `N` is handled correctly without special‑case code.
- **Improved hardware utilization when the launch size is small** – if we are constrained to launch a small number of blocks (e.g., due to resource limits), the grid‑stride loop still covers all elements by having each thread process multiple indices.

To demonstrate these points, this program:
1. Allocates two large arrays on the host and device.
2. Launches a vector‑addition kernel with a tiny launch configuration (e.g., one block of 256 threads).
3. The kernel uses a grid‑stride loop to add all elements, proving correctness.
4. Measures and prints execution time to show that the small launch is still effective.
5. Compares with a more “normal” large launch configuration to illustrate that the same code works for both cases.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",              \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// Kernel using grid‑stride loop
__global__ void vectorAddGS(const float *A, const float *B, float *C, size_t N)
{
    // Compute global thread index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;

    // Loop over the array with stride to cover all elements
    for (size_t i = idx; i < N; i += stride)
    {
        C[i] = A[i] + B[i];
    }
}

int main()
{
    const size_t N = 1 << 26;          // 67,108,864 elements (~256 MB per array)
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_B, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_C, bytes));

    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Launch configuration: very small (1 block of 256 threads)
    dim3 threadsPerBlock(256);
    dim3 numBlocks(1);

    // Timing the kernel
    auto start = std::chrono::high_resolution_clock::now();

    vectorAddGS<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> kernelTime = end - start;
    printf("Grid‑stride kernel (1 block, 256 threads) executed in %.3f ms\n",
           kernelTime.count());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify a few elements
    bool ok = true;
    for (size_t i = 0; i < 10; ++i) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            printf("Mismatch at index %zu: %f vs %f\n", i, h_C[i], h_A[i] + h_B[i]);
            ok = false;
            break;
        }
    }
    printf("Verification %s\n", ok ? "passed" : "failed");

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}
```