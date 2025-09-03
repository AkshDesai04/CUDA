/*
Aim of the program: Use `cudaEvent`s to time only the kernel execution time.

My thinking:
- The user wants a CUDA program that demonstrates how to use cudaEvent to measure the time taken by a kernel launch only.
- I'll implement a simple vector addition kernel as an example, since it's straightforward and common.
- Steps:
  1. Allocate host memory for two input arrays and one output array.
  2. Allocate device memory and copy inputs to device.
  3. Create two cudaEvent objects: `start` and `stop`.
  4. Record the `start` event.
  5. Launch the kernel.
  6. Record the `stop` event immediately after the launch.
  7. Synchronize on the `stop` event to ensure the kernel has finished.
  8. Calculate elapsed time using `cudaEventElapsedTime`.
  9. Copy the result back to host and optionally verify correctness.
 10. Clean up resources.
- I'll add error checking after each CUDA API call for robustness.
- The kernel itself will simply add two vectors element-wise.
- The code will be self‑contained in a single .cu file and can be compiled with `nvcc`.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Simple vector addition kernel
__global__ void vectorAdd(const float *A, const float *B, float *C, int N)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < N)
        C[idx] = A[idx] + B[idx];
}

// Helper macro for CUDA error checking
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", \
                __FILE__, __LINE__, err, cudaGetErrorName(err), cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }

int main(void)
{
    const int N = 1 << 20;                // 1M elements
    size_t size = N * sizeof(float);

    // Host allocation
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Device allocation
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size));

    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Define kernel launch configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Record the start event
    CUDA_CHECK(cudaEventRecord(start, 0));

    // Launch kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Record the stop event
    CUDA_CHECK(cudaEventRecord(stop, 0));

    // Wait for the stop event to complete
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Calculate elapsed time in milliseconds
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Kernel execution time: %f ms\n", milliseconds);

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Optional verification
    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            ok = false;
            printf("Verification failed at index %d: %f != %f + %f\n", i, h_C[i], h_A[i], h_B[i]);
            break;
        }
    }
    if (ok) printf("Result verification: PASSED\n");

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
