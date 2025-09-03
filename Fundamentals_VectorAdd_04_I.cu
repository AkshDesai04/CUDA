```cpp
/*
Aim of the program:
Time the subtraction kernel. Is it significantly different from addition?

Thinking:
To answer the question we need to benchmark two simple element‑wise kernels: one that
adds two vectors and one that subtracts them.  CUDA timing is best done with
cudaEvent_t which gives sub‑microsecond resolution.  We will:

1. Allocate two large vectors on the host and fill them with deterministic data.
2. Allocate corresponding device buffers and copy the data to the GPU.
3. Launch an addition kernel and time it with events.
4. Launch a subtraction kernel (same size, same launch configuration) and time it.
5. Copy the results back and optionally verify correctness.
6. Print the elapsed times and the ratio to see if subtraction is noticeably
   slower or faster than addition.

Both kernels are trivially identical except for the arithmetic operator,
so any performance difference should stem from compiler optimizations or the
GPU’s arithmetic unit handling of the two operations.  The program is
self‑contained and can be compiled with `nvcc`.

*/

#include <iostream>
#include <iomanip>
#include <cuda_runtime.h>

#define N (1<<20)          // 1M elements
#define THREADS_PER_BLOCK 256

__global__ void addKernel(const float* a, const float* b, float* c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

__global__ void subKernel(const float* a, const float* b, float* c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] - b[idx];
}

void checkCudaError(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << msg << " : " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main()
{
    // Host allocations
    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_c_add = new float[N];
    float *h_c_sub = new float[N];

    // Initialize data
    for (int i = 0; i < N; ++i)
    {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(N - i);
    }

    // Device allocations
    float *d_a, *d_b, *d_c_add, *d_c_sub;
    checkCudaError(cudaMalloc((void**)&d_a, N * sizeof(float)), "malloc d_a");
    checkCudaError(cudaMalloc((void**)&d_b, N * sizeof(float)), "malloc d_b");
    checkCudaError(cudaMalloc((void**)&d_c_add, N * sizeof(float)), "malloc d_c_add");
    checkCudaError(cudaMalloc((void**)&d_c_sub, N * sizeof(float)), "malloc d_c_sub");

    // Copy inputs to device
    checkCudaError(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice), "memcpy h_a->d_a");
    checkCudaError(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice), "memcpy h_b->d_b");

    // Timing events
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "event create start");
    checkCudaError(cudaEventCreate(&stop), "event create stop");

    dim3 blockSize(THREADS_PER_BLOCK);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    // Add kernel timing
    checkCudaError(cudaEventRecord(start), "record start add");
    addKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c_add, N);
    checkCudaError(cudaEventRecord(stop), "record stop add");
    checkCudaError(cudaEventSynchronize(stop), "sync stop add");
    float ms_add = 0;
    checkCudaError(cudaEventElapsedTime(&ms_add, start, stop), "elapsed add");

    // Sub kernel timing
    checkCudaError(cudaEventRecord(start), "record start sub");
    subKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c_sub, N);
    checkCudaError(cudaEventRecord(stop), "record stop sub");
    checkCudaError(cudaEventSynchronize(stop), "sync stop sub");
    float ms_sub = 0;
    checkCudaError(cudaEventElapsedTime(&ms_sub, start, stop), "elapsed sub");

    // Copy results back
    checkCudaError(cudaMemcpy(h_c_add, d_c_add, N * sizeof(float), cudaMemcpyDeviceToHost), "memcpy d_c_add->h_c_add");
    checkCudaError(cudaMemcpy(h_c_sub, d_c_sub, N * sizeof(float), cudaMemcpyDeviceToHost), "memcpy d_c_sub->h_c_sub");

    // Verify a few results (optional)
    bool correct = true;
    for (int i = 0; i < 10 && correct; ++i)
    {
        if (fabs(h_c_add[i] - (h_a[i] + h_b[i])) > 1e-5f) correct = false;
        if (fabs(h_c_sub[i] - (h_a[i] - h_b[i])) > 1e-5f) correct = false;
    }
    std::cout << "Verification: " << (correct ? "PASS" : "FAIL") << std::endl;

    // Output timings
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Addition kernel time:  " << ms_add << " ms" << std::endl;
    std::cout << "Subtraction kernel time: " << ms_sub << " ms" << std::endl;
    std::cout << "Speed ratio (sub/add):  " << ms_sub / ms_add << std::endl;

    // Clean up
    delete[] h_a; delete[] h_b; delete[] h_c_add; delete[] h_c_sub;
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c_add); cudaFree(d_c_sub);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return 0;
}
```