```cpp
/*
Aim of the program:
Is there a performance difference between a standard launch (one thread per element) and a grid-stride launch (many elements per thread) for a problem that fits in one launch? Time both.

Thinking:
To answer the question we create a simple vector operation (c[i] = a[i] * b[i] + c[i]) on a large array that can fit into a single kernel launch. We measure the elapsed GPU time for two kernel implementations:

1. Standard launch: one thread per element. The kernel writes to its corresponding element only.
2. Grid-stride launch: a single thread processes multiple elements via a stride loop. The kernel still uses the same grid dimensions as the standard launch; each thread iterates over elements spaced by the grid stride.

We will:
- Allocate large float arrays (size 16,777,216 ≈ 16M) on host and device.
- Initialize a and b with random values; c with zeros.
- Run the standard kernel, time it with CUDA events, copy back results, and verify correctness.
- Reinitialize c to zeros, run the grid‑stride kernel, time it, copy back results, and verify correctness.
- Print elapsed times for comparison.

The program uses CUDA error checking, CUDA events for precise timing, and prints the timing results to the console. It is self‑contained and can be compiled with `nvcc` as a `.cu` file.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <ctime>

// Error checking macro
#define CUDA_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        std::cerr << "CUDA Error: " << cudaGetErrorString(code) << " " << file << ":" << line << std::endl;
        if (abort) exit(code);
    }
}

// Standard kernel: one thread per element
__global__ void stdKernel(const float *a, const float *b, float *c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        c[idx] = a[idx] * b[idx] + c[idx];
    }
}

// Grid-stride kernel: each thread processes multiple elements
__global__ void gsKernel(const float *a, const float *b, float *c, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < N; i += stride)
    {
        c[i] = a[i] * b[i] + c[i];
    }
}

// Simple checksum to verify correctness
float computeChecksum(const float *data, int N)
{
    double sum = 0.0;
    for (int i = 0; i < N; ++i)
        sum += data[i];
    return static_cast<float>(sum);
}

int main()
{
    const int N = 16777216;          // 16M elements
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;

    size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_c = new float[N];

    // Initialize host arrays
    std::srand(static_cast<unsigned>(std::time(nullptr)));
    for (int i = 0; i < N; ++i)
    {
        h_a[i] = static_cast<float>(std::rand()) / RAND_MAX;
        h_b[i] = static_cast<float>(std::rand()) / RAND_MAX;
        h_c[i] = 0.0f;
    }

    // Allocate device memory
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc((void**)&d_a, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_b, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_c, bytes));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, h_c, bytes, cudaMemcpyHostToDevice));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // ----------------- Standard Launch -----------------
    CUDA_CHECK(cudaEventRecord(start, 0));
    stdKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float msStd = 0;
    CUDA_CHECK(cudaEventElapsedTime(&msStd, start, stop));

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    float checksumStd = computeChecksum(h_c, N);

    // Reinitialize c to zeros for next run
    CUDA_CHECK(cudaMemset(d_c, 0, bytes));

    // ----------------- Grid-Stride Launch -----------------
    CUDA_CHECK(cudaEventRecord(start, 0));
    gsKernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float msGS = 0;
    CUDA_CHECK(cudaEventElapsedTime(&msGS, start, stop));

    // Copy result back
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));
    float checksumGS = computeChecksum(h_c, N);

    // ----------------- Results -----------------
    std::cout << "Standard launch time: " << msStd << " ms, checksum = " << checksumStd << std::endl;
    std::cout << "Grid-stride launch time: " << msGS << " ms, checksum = " << checksumGS << std::endl;

    // Cleanup
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
```