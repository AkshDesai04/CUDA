```cu
/*
Aim of the program:
Perform the comparison for a more compute-intensive kernel, like `sin(cos(x))`. Does the GPU speedup increase?

Thinking:
To evaluate whether a more compute-intensive kernel yields a greater GPU speedup, we implement two versions of the same operation:
1. CPU implementation using a simple loop.
2. GPU implementation as a CUDA kernel where each thread computes sin(cos(x)) for a distinct element.

We choose a large problem size (e.g., 10 million elements) to better expose parallelism and kernel launch overhead. 
The data type chosen is double, but for better GPU performance double precision might be slower on some devices; however we keep double to demonstrate the computation cost. 
We time each implementation:
- For GPU: use cudaEventRecord for start/end, cudaMemcpy for data transfer.
- For CPU: use std::chrono high_resolution_clock.

After execution, we output the time taken by CPU and GPU (including data transfer) and compute the speedup factor (CPU time / GPU time). 
If the speedup is higher for the sin(cos(x)) kernel than a simpler kernel (like addition), it indicates that GPU benefit scales with computational intensity.

We also check for CUDA errors and ensure proper cleanup. The code is self‑contained and can be compiled with nvcc.
*/

#include <cstdio>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>

// CUDA kernel: compute sin(cos(x)) for each element
__global__ void sin_cos_kernel(double* out, const double* in, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        double val = in[idx];
        out[idx] = sin(cos(val));
    }
}

// Helper to check CUDA errors
void checkCudaError(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    const int N = 10'000'000; // 10 million elements
    size_t bytes = N * sizeof(double);

    // Allocate host memory
    double* h_in = new double[N];
    double* h_out_cpu = new double[N];
    double* h_out_gpu = new double[N];

    // Initialize input with some values
    for (int i = 0; i < N; ++i) {
        h_in[i] = static_cast<double>(i) * 0.0001; // arbitrary values
    }

    // CPU computation
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; ++i) {
        double val = h_in[i];
        h_out_cpu[i] = sin(cos(val));
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = cpu_end - cpu_start;
    printf("CPU time: %.6f seconds\n", cpu_duration.count());

    // GPU computation
    double *d_in = nullptr, *d_out = nullptr;
    checkCudaError(cudaMalloc((void**)&d_in, bytes), "cudaMalloc d_in");
    checkCudaError(cudaMalloc((void**)&d_out, bytes), "cudaMalloc d_out");

    // Copy input to device
    checkCudaError(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice), "cudaMemcpy H2D");

    // Setup kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Timing GPU (including transfer)
    cudaEvent_t start, stop;
    checkCudaError(cudaEventCreate(&start), "cudaEventCreate start");
    checkCudaError(cudaEventCreate(&stop), "cudaEventCreate stop");

    checkCudaError(cudaEventRecord(start, 0), "cudaEventRecord start");

    sin_cos_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_out, d_in, N);
    checkCudaError(cudaGetLastError(), "Kernel launch");

    // Copy result back to host
    checkCudaError(cudaMemcpy(h_out_gpu, d_out, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");

    checkCudaError(cudaEventRecord(stop, 0), "cudaEventRecord stop");
    checkCudaError(cudaEventSynchronize(stop), "cudaEventSynchronize stop");

    float milliseconds = 0;
    checkCudaError(cudaEventElapsedTime(&milliseconds, start, stop), "cudaEventElapsedTime");
    double gpu_duration = milliseconds / 1000.0;
    printf("GPU time (including transfer): %.6f seconds\n", gpu_duration);

    // Compute speedup
    double speedup = cpu_duration.count() / gpu_duration;
    printf("Speedup: %.2fx\n", speedup);

    // Optional: Verify correctness
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_out_cpu[i] - h_out_gpu[i]) > 1e-12) {
            ok = false;
            printf("Mismatch at index %d: CPU %f vs GPU %f\n", i, h_out_cpu[i], h_out_gpu[i]);
            break;
        }
    }
    if (ok) {
        printf("Result verification: PASS\n");
    } else {
        printf("Result verification: FAIL\n");
    }

    // Cleanup
    delete[] h_in;
    delete[] h_out_cpu;
    delete[] h_out_gpu;
    checkCudaError(cudaFree(d_in), "cudaFree d_in");
    checkCudaError(cudaFree(d_out), "cudaFree d_out");
    checkCudaError(cudaEventDestroy(start), "cudaEventDestroy start");
    checkCudaError(cudaEventDestroy(stop), "cudaEventDestroy stop");

    return 0;
}
```