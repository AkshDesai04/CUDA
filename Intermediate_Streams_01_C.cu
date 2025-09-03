```cuda
/*
 * Aim: A `cudaDeviceSynchronize()` is implicitly happening after each operation, making it fully synchronous. Time the total execution of the loop.
 *
 * Thought process:
 * 1. We need a minimal CUDA program that runs a loop of operations that are normally asynchronous but, due to implicit synchronization points (e.g., cudaMemcpy, kernel launch), become synchronous.
 * 2. The loop will perform a simple vector addition: copy data to device, launch kernel, copy result back.
 * 3. Each cudaMemcpy from device to host is synchronous, and kernel launches are asynchronous but will be implicitly synchronized by the subsequent memcpy.
 * 4. To measure the total execution time of the loop, we use CUDA events (`cudaEventRecord` and `cudaEventElapsedTime`) which give us time in milliseconds on the GPU.
 * 5. The program will allocate host and device memory, initialize data, run the loop, record timing, and print the elapsed time.
 * 6. Finally, we free allocated resources.
 *
 * This demonstrates that despite launching kernels asynchronously, the implicit synchronization points cause the loop to behave synchronously, and we can accurately time the total duration.
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024            // Size of vectors
#define LOOP_ITERATIONS 1000  // Number of times to repeat the operation

// Simple vector addition kernel
__global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    // Host allocations
    float *h_A = (float*)malloc(N * sizeof(float));
    float *h_B = (float*)malloc(N * sizeof(float));
    float *h_C = (float*)malloc(N * sizeof(float));

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(2 * i);
    }

    // Device allocations
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, N * sizeof(float));
    cudaMalloc((void**)&d_B, N * sizeof(float));
    cudaMalloc((void**)&d_C, N * sizeof(float));

    // Timing events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start, 0);

    // Loop performing the vector addition N times
    for (int iter = 0; iter < LOOP_ITERATIONS; ++iter) {
        // Copy data to device (synchronous for this simple example)
        cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice);

        // Launch kernel
        int threadsPerBlock = 256;
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
        vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

        // Copy result back to host (synchronizes with kernel)
        cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost);
        // At this point, cudaMemcpy acts as an implicit cudaDeviceSynchronize,
        // making each iteration fully synchronous.
    }

    // Record stop event
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Total time for %d iterations: %f ms\n", LOOP_ITERATIONS, milliseconds);

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```