```cpp
/*
If you launch more threads than elements N, the grid-stride loop will have some threads do no work. Verify this is handled correctly.

Thinking:
- The goal is to demonstrate that launching a kernel with more threads than the number of elements (N) does not cause out-of-bounds accesses or incorrect results.
- We use a grid-stride loop inside the kernel so that each thread can process multiple elements if needed. In this test, we intentionally launch more threads than elements so that many threads will have nothing to process.
- To verify correctness, we:
  1. Allocate an array of N floats and initialize it with known values.
  2. Compute the expected sum on the host.
  3. Launch a kernel that sums the array into a single float using atomicAdd.
  4. Count how many threads actually performed work using an atomic counter (`activeThreadCount`). Threads with an index >= N will skip the loop entirely and should not increment this counter.
  5. After kernel execution, copy back the sum and the counter.
  6. Compare the GPU-computed sum with the host sum to ensure correctness.
  7. Print the number of active threads. In this scenario, the number should equal N, confirming that exactly N threads did work and the remaining threads did none.
- The program includes error checking for CUDA API calls.
- The kernel uses `atomicAdd` on a float (supported on compute capability >= 2.0) to avoid race conditions.
- We deliberately set `N` (e.g., 100) and launch more threads (e.g., 128) to enforce the situation where some threads do nothing.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>

// CUDA error checking macro
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                  \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Kernel that sums array elements using a grid-stride loop
// It also counts how many threads performed at least one iteration
__global__ void sumArrayKernel(const float *d_in, float *d_out, int N, int *d_activeThreadCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int gridSize = gridDim.x * blockDim.x;

    bool didWork = false;

    // Grid-stride loop: each thread processes multiple elements if needed
    for (int i = idx; i < N; i += gridSize)
    {
        // Perform atomic addition to the global sum
        atomicAdd(d_out, d_in[i]);
        didWork = true;
    }

    // If this thread processed at least one element, increment active thread counter
    if (didWork)
    {
        atomicAdd(d_activeThreadCount, 1);
    }
}

int main()
{
    const int N = 100;               // Number of array elements
    const int threadsPerBlock = 128; // More threads than N
    const int blocks = 1;            // Single block

    // Allocate host memory and initialize array
    float *h_in = (float*)malloc(N * sizeof(float));
    if (!h_in) {
        std::cerr << "Failed to allocate host memory\n";
        return EXIT_FAILURE;
    }

    for (int i = 0; i < N; ++i)
    {
        h_in[i] = static_cast<float>(i + 1); // 1.0, 2.0, ..., N
    }

    // Compute expected sum on host
    float hostSum = 0.0f;
    for (int i = 0; i < N; ++i)
    {
        hostSum += h_in[i];
    }

    // Allocate device memory
    float *d_in = nullptr;
    float *d_sum = nullptr;
    int *d_activeCount = nullptr;

    CUDA_CHECK(cudaMalloc((void**)&d_in, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_sum, sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_activeCount, sizeof(int)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_sum, 0, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_activeCount, 0, sizeof(int)));

    // Launch kernel
    sumArrayKernel<<<blocks, threadsPerBlock>>>(d_in, d_sum, N, d_activeCount);
    CUDA_CHECK(cudaGetLastError());

    // Copy results back to host
    float gpuSum = 0.0f;
    int activeThreadCount = 0;
    CUDA_CHECK(cudaMemcpy(&gpuSum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&activeThreadCount, d_activeCount, sizeof(int), cudaMemcpyDeviceToHost));

    // Verify results
    std::cout << "Host computed sum: " << hostSum << std::endl;
    std::cout << "GPU computed sum:  " << gpuSum << std::endl;
    std::cout << "Active threads:    " << activeThreadCount << std::endl;
    std::cout << "Expected active threads (should equal N): " << N << std::endl;

    if (abs(gpuSum - hostSum) < 1e-5f && activeThreadCount == N)
    {
        std::cout << "Verification PASSED: All threads handled correctly.\n";
    }
    else
    {
        std::cout << "Verification FAILED: Discrepancies found.\n";
    }

    // Clean up
    free(h_in);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_sum));
    CUDA_CHECK(cudaFree(d_activeCount));

    return 0;
}
```