```cpp
/*
Aim of the program:
Compare the performance of `cudaMemset` vs. your custom zeroing kernel for a very large array. `cudaMemset` is usually highly optimized.

Thinking:
The goal is to create a CUDA program that allocates a large array on the GPU, zeros it using two different methods (cudaMemset and a custom kernel), measures the execution time of each method, and verifies correctness. To make the comparison realistic, the array will be large enough to fully occupy the GPU memory but not exceed typical device capacities. A common size is several hundred megabytes of data; here I chose 256 M integers (~1 GB) which is within the limits of many modern GPUs with 8 GB+ memory.

Key design decisions:
1. Use `int` data type for simplicity. The array size is expressed in number of elements (`N`) and in bytes (`size_in_bytes`).
2. Implement a straightforward CUDA kernel `zeroKernel` that writes zero to each array element. Each thread handles one element, using global index calculation.
3. For timing, use CUDA events (`cudaEventCreate`, `cudaEventRecord`, `cudaEventElapsedTime`). This provides millisecond precision and accounts only for GPU work, excluding CPU overhead.
4. After each zeroing method, copy the array back to host memory with `cudaMemcpy` and validate that every element is zero. This ensures that the operations are correct.
5. Include robust error checking for all CUDA API calls via a helper macro `CHECK_CUDA_ERR`.
6. Output the measured times and compute speedup of the kernel relative to `cudaMemset`.
7. Keep the code C++-compatible (`extern "C"` not required) and fully self‑contained; it compiles with `nvcc` and produces a single `.cu` file.

The program follows these steps:
- Parse optional command line argument for array size; otherwise use default 256 M.
- Allocate device memory.
- Run `cudaMemset`, time it, copy back, and validate.
- Run custom kernel, time it, copy back, and validate.
- Print timings and speedup.
- Free memory and exit.

Below is the complete source code. Compile with: `nvcc -O2 -arch=sm_70 zero_compare.cu -o zero_compare`
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA_ERR(call)                                     \
    do {                                                         \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess) {                                \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",         \
                    __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

__global__ void zeroKernel(int *data, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = 0;
    }
}

void verifyZero(const int *hostArray, size_t N)
{
    for (size_t i = 0; i < N; ++i) {
        if (hostArray[i] != 0) {
            fprintf(stderr, "Verification failed at index %zu: value %d\n", i, hostArray[i]);
            exit(EXIT_FAILURE);
        }
    }
}

int main(int argc, char *argv[])
{
    // Default array size: 256 million integers (~1 GB)
    size_t N = 256ULL * 1024ULL * 1024ULL; // 256M elements
    if (argc > 1) {
        N = strtoull(argv[1], NULL, 10);
    }
    size_t size_in_bytes = N * sizeof(int);
    printf("Array size: %zu elements (%zu bytes)\n", N, size_in_bytes);

    // Allocate device memory
    int *d_array = NULL;
    CHECK_CUDA_ERR(cudaMalloc((void **)&d_array, size_in_bytes));

    // Allocate host memory for verification
    int *h_array = (int *)malloc(size_in_bytes);
    if (!h_array) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        exit(EXIT_FAILURE);
    }

    // CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA_ERR(cudaEventCreate(&start));
    CHECK_CUDA_ERR(cudaEventCreate(&stop));

    // ---- 1) cudaMemset ----
    CHECK_CUDA_ERR(cudaEventRecord(start, 0));
    CHECK_CUDA_ERR(cudaMemset(d_array, 0, size_in_bytes));
    CHECK_CUDA_ERR(cudaEventRecord(stop, 0));
    CHECK_CUDA_ERR(cudaEventSynchronize(stop));

    float msMemset = 0.0f;
    CHECK_CUDA_ERR(cudaEventElapsedTime(&msMemset, start, stop));
    printf("cudaMemset time: %.3f ms\n", msMemset);

    // Copy back and verify
    CHECK_CUDA_ERR(cudaMemcpy(h_array, d_array, size_in_bytes, cudaMemcpyDeviceToHost));
    verifyZero(h_array, N);
    printf("Verification after cudaMemset succeeded.\n");

    // ---- 2) Custom zeroing kernel ----
    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    CHECK_CUDA_ERR(cudaEventRecord(start, 0));
    zeroKernel<<<gridSize, blockSize>>>(d_array, N);
    CHECK_CUDA_ERR(cudaGetLastError()); // Check kernel launch
    CHECK_CUDA_ERR(cudaEventRecord(stop, 0));
    CHECK_CUDA_ERR(cudaEventSynchronize(stop));

    float msKernel = 0.0f;
    CHECK_CUDA_ERR(cudaEventElapsedTime(&msKernel, start, stop));
    printf("Custom kernel time: %.3f ms\n", msKernel);

    // Copy back and verify
    CHECK_CUDA_ERR(cudaMemcpy(h_array, d_array, size_in_bytes, cudaMemcpyDeviceToHost));
    verifyZero(h_array, N);
    printf("Verification after custom kernel succeeded.\n");

    // Speedup calculation
    if (msKernel > 0.0f) {
        float speedup = msMemset / msKernel;
        printf("Speedup (cudaMemset / kernel): %.2f×\n", speedup);
    }

    // Clean up
    free(h_array);
    CHECK_CUDA_ERR(cudaFree(d_array));
    CHECK_CUDA_ERR(cudaEventDestroy(start));
    CHECK_CUDA_ERR(cudaEventDestroy(stop));

    return 0;
}
```