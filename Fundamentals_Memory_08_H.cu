/*
Ensure you call `cudaEventDestroy` to clean up the event handles.
The aim of this program is to demonstrate the proper creation, usage, and destruction of CUDA event handles for timing a simple vector addition kernel. I chose a straightforward vector addition as it provides a clear workload that can be parallelized, allows us to measure GPU execution time accurately with events, and showcases the necessity of cleaning up resources. The program allocates host pinned memory for efficient transfers, copies data to the device, launches the kernel, records timing events before and after the kernel launch, synchronizes, retrieves elapsed time, and finally frees all allocated memory and destroys the CUDA event objects with `cudaEventDestroy`. This illustrates the complete lifecycle of event usage, ensuring no resource leaks occur.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Error checking macro
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n",
                cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Simple vector addition kernel
__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

int main(void) {
    const int N = 1 << 20;               // 1 million elements
    const size_t size = N * sizeof(float);

    // Allocate pinned host memory
    float *h_A, *h_B, *h_C;
    cudaCheckError(cudaMallocHost(&h_A, size));
    cudaCheckError(cudaMallocHost(&h_B, size));
    cudaCheckError(cudaMallocHost(&h_C, size));

    // Initialize host data
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(i * 2);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaCheckError(cudaMalloc(&d_A, size));
    cudaCheckError(cudaMalloc(&d_B, size));
    cudaCheckError(cudaMalloc(&d_C, size));

    // Copy data to device
    cudaCheckError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Kernel launch configuration
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaCheckError(cudaEventCreate(&start));
    cudaCheckError(cudaEventCreate(&stop));

    // Record start event
    cudaCheckError(cudaEventRecord(start, 0));

    // Launch kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaCheckError(cudaGetLastError());

    // Record stop event
    cudaCheckError(cudaEventRecord(stop, 0));
    cudaCheckError(cudaEventSynchronize(stop));

    // Compute elapsed time
    float milliseconds = 0.0f;
    cudaCheckError(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Kernel execution time: %f ms\n", milliseconds);

    // Copy result back to host
    cudaCheckError(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Cleanup: free device memory
    cudaCheckError(cudaFree(d_A));
    cudaCheckError(cudaFree(d_B));
    cudaCheckError(cudaFree(d_C));

    // Cleanup: free host memory
    cudaCheckError(cudaFreeHost(h_A));
    cudaCheckError(cudaFreeHost(h_B));
    cudaCheckError(cudaFreeHost(h_C));

    // Destroy CUDA events to release resources
    cudaCheckError(cudaEventDestroy(start));
    cudaCheckError(cudaEventDestroy(stop));

    return 0;
}
