/*
Use `nsight compute` or `nvprof` to measure the global memory bandwidth and transaction count for the naive kernel. Note the high number of transactions.

Thinking:
The goal is to provide a simple CUDA program that launches a naive global‑memory kernel, so that a user can run external profiling tools such as Nsight Compute or nvprof to inspect the global memory bandwidth and the number of memory transactions. The kernel performs a straightforward element‑wise operation (here, multiplying each element by 2.0f). Because each thread accesses a separate element of the input array and writes to the output array, the memory accesses are contiguous but still involve a large amount of global memory traffic. When profiling this kernel on a real GPU, you will see a high number of transactions and a measurable bandwidth usage. The program allocates large arrays to ensure the statistics are meaningful, times the kernel, and cleans up resources. The user can compile with `nvcc` and then profile with:

    nvprof -f ./naive_mem_bandwidth
    or
    nsight compute ./naive_mem_bandwidth

The comment above contains the exact prompt text verbatim, as required.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>

#define N (1 << 24)           // 16 million elements (~64 MB for float arrays)
#define BLOCK_SIZE 256        // Number of threads per block

__global__ void naiveKernel(const float *in, float *out, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Simple element‑wise multiplication; accesses global memory twice
        out[idx] = in[idx] * 2.0f;
    }
}

int main()
{
    float *h_in  = new float[N];
    float *h_out = new float[N];

    // Initialize host data
    for (int i = 0; i < N; ++i) {
        h_in[i] = static_cast<float>(i);
    }

    // Device pointers
    float *d_in  = nullptr;
    float *d_out = nullptr;

    size_t bytes = N * sizeof(float);

    // Allocate device memory
    cudaMalloc((void **)&d_in,  bytes);
    cudaMalloc((void **)&d_out, bytes);

    // Copy input data to device
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    // Configure launch parameters
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE);

    // Create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start, 0);

    // Launch naive kernel
    naiveKernel<<<grid, block>>>(d_in, d_out, N);

    // Record stop event
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    // Copy result back to host (optional)
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    // Verify a few results
    bool ok = true;
    for (int i = 0; i < 10; ++i) {
        if (h_out[i] != h_in[i] * 2.0f) {
            ok = false;
            break;
        }
    }
    std::cout << "Verification: " << (ok ? "PASSED" : "FAILED") << std::endl;

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Kernel time: " << ms << " ms" << std::endl;
    std::cout << "Global memory traffic: " << (2 * bytes) / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Throughput: " << (2 * bytes) / (ms / 1000.0) / (1024.0 * 1024.0) << " MB/s" << std::endl;

    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out);
    delete[] h_in;
    delete[] h_out;

    return 0;
}
