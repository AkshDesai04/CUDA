/*
Aim: Use your GpuTimer class to time a vector addition kernel.

Thinking:
1. The task requires a CUDA program that times a vector addition kernel using a GpuTimer class. Since GpuTimer is not part of the CUDA runtime, I'll implement a small utility class that uses CUDA events to measure elapsed GPU time. The class will have methods start(), stop(), elapsed(), and reset().

2. The vector addition kernel will add two float arrays element‑wise. It will be launched with a typical grid/block configuration.

3. In main:
   - Allocate host vectors A, B, and C with a chosen size (e.g., 1e7 elements).
   - Initialize A and B with some data.
   - Allocate corresponding device memory.
   - Copy A and B to the device.
   - Create a GpuTimer instance.
   - Start the timer, launch the kernel, synchronize, stop the timer.
   - Copy result back to host and optionally verify correctness.
   - Print the elapsed time.

4. The code will be self‑contained in a single .cu file and will compile with `nvcc`.

5. Edge cases: error checking for CUDA calls, ensuring the timer works correctly. We'll also ensure we clean up all resources.

Now the actual CUDA code follows.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// GpuTimer class using CUDA events
class GpuTimer {
public:
    GpuTimer() {
        cudaEventCreate(&startEvent);
        cudaEventCreate(&stopEvent);
    }

    ~GpuTimer() {
        cudaEventDestroy(startEvent);
        cudaEventDestroy(stopEvent);
    }

    void reset() {
        cudaEventRecord(startEvent, 0);
        cudaEventRecord(stopEvent, 0);
    }

    void start() {
        cudaEventRecord(startEvent, 0);
    }

    void stop() {
        cudaEventRecord(stopEvent, 0);
        cudaEventSynchronize(stopEvent);
    }

    float elapsed() const { // milliseconds
        float ms = 0;
        cudaEventElapsedTime(&ms, startEvent, stopEvent);
        return ms;
    }

private:
    cudaEvent_t startEvent, stopEvent;
};

// Vector addition kernel
__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    const int N = 1 << 24; // about 16 million elements
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(N - i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);

    // Copy data to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Determine grid/block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Create and use GpuTimer
    GpuTimer timer;
    timer.start();

    // Launch kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Stop timer after kernel execution
    timer.stop();

    // Copy result back to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Verify a few elements
    bool correct = true;
    for (int i = 0; i < 10; ++i) {
        if (fabs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5) {
            correct = false;
            break;
        }
    }

    // Print results
    printf("Vector addition completed. Elapsed time: %f ms\n", timer.elapsed());
    printf("Verification: %s\n", correct ? "PASSED" : "FAILED");

    // Clean up
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
