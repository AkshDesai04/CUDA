```cuda
/*
Analyze shared memory bank conflicts. Write a kernel where threads access shared memory with a stride of 1 (`shared_mem[threadIdx.x]`) and another where they access with a stride equal to the number of banks (e.g., 32, `shared_mem[32 * threadIdx.x]`). Profile both and explain the performance difference.

Thinking:
- Shared memory on a CUDA device is divided into 32 banks (on most current GPUs). Each bank can service one access per cycle.
- If multiple threads in a warp access the same bank (e.g., consecutive addresses for stride 1), the accesses serialize, causing bank conflicts and slowing the kernel.
- If each thread accesses a distinct bank (e.g., stride equal to the bank count, 32), accesses can occur in parallel, avoiding conflicts.
- We'll implement two kernels: one with stride 1 and one with stride 32. Both will perform a simple load from global memory into shared memory, then write back to global memory.
- We will measure execution times of both kernels using CUDA events to illustrate the performance difference.
- The expected result is that the stride-32 kernel runs faster because it avoids bank conflicts.
*/

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>

// Kernel with stride 1 (potential bank conflicts)
__global__ void stride1Kernel(const float *d_in, float *d_out)
{
    // Static shared memory sized to blockDim.x elements
    __shared__ float shared_mem[256]; // Assuming blockDim.x <= 256

    int tid = threadIdx.x;
    // Load from global memory into shared memory
    shared_mem[tid] = d_in[blockIdx.x * blockDim.x + tid];
    __syncthreads();

    // Simple read/write to verify correctness
    float val = shared_mem[tid];
    d_out[blockIdx.x * blockDim.x + tid] = val;
}

// Kernel with stride equal to number of banks (32) to avoid conflicts
__global__ void stride32Kernel(const float *d_in, float *d_out)
{
    // Dynamic shared memory; size will be set when launching kernel
    extern __shared__ float shared_mem[];

    int tid = threadIdx.x;
    // Access with stride 32 to map each thread to a distinct bank
    shared_mem[32 * tid] = d_in[blockIdx.x * blockDim.x + tid];
    __syncthreads();

    float val = shared_mem[32 * tid];
    d_out[blockIdx.x * blockDim.x + tid] = val;
}

int main()
{
    const int threadsPerBlock = 256;
    const int blocksPerGrid = 64; // Total elements = 64 * 256 = 16384
    const int totalElements = threadsPerBlock * blocksPerGrid;

    // Allocate host memory
    float *h_in = (float*)malloc(totalElements * sizeof(float));
    float *h_out1 = (float*)malloc(totalElements * sizeof(float));
    float *h_out2 = (float*)malloc(totalElements * sizeof(float));

    // Initialize input with random data
    srand((unsigned)time(NULL));
    for (int i = 0; i < totalElements; ++i)
        h_in[i] = static_cast<float>(rand()) / RAND_MAX;

    // Allocate device memory
    float *d_in, *d_out1, *d_out2;
    cudaMalloc((void**)&d_in,    totalElements * sizeof(float));
    cudaMalloc((void**)&d_out1,  totalElements * sizeof(float));
    cudaMalloc((void**)&d_out2,  totalElements * sizeof(float));

    // Copy input to device
    cudaMemcpy(d_in, h_in, totalElements * sizeof(float), cudaMemcpyHostToDevice);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // -----------------------
    // Launch stride1 kernel
    // -----------------------
    cudaEventRecord(start);
    stride1Kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out1);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float timeStride1 = 0.0f;
    cudaEventElapsedTime(&timeStride1, start, stop);

    // -----------------------
    // Launch stride32 kernel
    // -----------------------
    // Need to provide shared memory size: 32 * threadsPerBlock * sizeof(float)
    size_t sharedMemSize = 32 * threadsPerBlock * sizeof(float);

    cudaEventRecord(start);
    stride32Kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_in, d_out2);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float timeStride32 = 0.0f;
    cudaEventElapsedTime(&timeStride32, start, stop);

    // Copy results back to host for validation (optional)
    cudaMemcpy(h_out1, d_out1, totalElements * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_out2, d_out2, totalElements * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify correctness
    bool correct = true;
    for (int i = 0; i < totalElements; ++i)
    {
        if (fabs(h_out1[i] - h_in[i]) > 1e-5 || fabs(h_out2[i] - h_in[i]) > 1e-5)
        {
            correct = false;
            break;
        }
    }

    std::cout << "Stride-1 kernel time:   " << timeStride1 << " ms\n";
    std::cout << "Stride-32 kernel time:  " << timeStride32 << " ms\n";
    std::cout << "Correctness: " << (correct ? "PASS" : "FAIL") << std::endl;

    // Explanation of performance difference
    std::cout << "\nPerformance Explanation:\n";
    std::cout << "The stride-1 kernel causes bank conflicts because consecutive threads access consecutive addresses in shared memory, mapping many threads to the same bank. This serialization reduces throughput.\n";
    std::cout << "The stride-32 kernel spaces accesses by the number of banks (32), ensuring each thread accesses a unique bank. As a result, accesses can proceed in parallel, avoiding serialization and yielding better performance.\n";

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_in);
    cudaFree(d_out1);
    cudaFree(d_out2);
    free(h_in);
    free(h_out1);
    free(h_out2);

    return 0;
}
```