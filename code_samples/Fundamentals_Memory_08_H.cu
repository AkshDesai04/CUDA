/*
Ensure you call cudaEventDestroy to clean up the event handles.
This program demonstrates the proper use of CUDA events for timing a simple
kernel launch and shows how to clean up event handles with cudaEventDestroy.
The goal is to illustrate that after we finish using cuda events we must
destroy them to avoid leaking GPU resources.  The program performs a
vector addition kernel, measures the elapsed time between event recordings,
prints the elapsed time, and finally destroys the event objects before
exiting.  The code includes basic error checking via a macro and is written
in standard C for CUDA.  It can be compiled with `nvcc` and run on any
CUDA-capable device.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(err)                                   \
    do {                                                  \
        cudaError_t err__ = (err);                        \
        if (err__ != cudaSuccess) {                      \
            fprintf(stderr,                               \
                    "CUDA error at %s:%d: %s\n",          \
                    __FILE__, __LINE__,                  \
                    cudaGetErrorString(err__));          \
            exit(EXIT_FAILURE);                           \
        }                                                 \
    } while (0)

// Simple vector addition kernel
__global__ void vectorAdd(const float *A, const float *B, float *C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        C[idx] = A[idx] + B[idx];
}

int main(void)
{
    const int N = 1 << 20;           // 1M elements
    const size_t size = N * sizeof(float);

    // Host allocation
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    // Initialize input data
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    // Device allocation
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void **)&d_A, size));
    CUDA_CHECK(cudaMalloc((void **)&d_B, size));
    CUDA_CHECK(cudaMalloc((void **)&d_C, size));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Record start event
    CUDA_CHECK(cudaEventRecord(start, 0));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Record stop event
    CUDA_CHECK(cudaEventRecord(stop, 0));

    // Wait for the stop event to complete
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Compute elapsed time
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Kernel execution time: %.3f ms\n", milliseconds);

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Verify a few results
    for (int i = 0; i < 5; ++i) {
        printf("C[%d] = %.1f (expected %.1f)\n", i, h_C[i], h_A[i] + h_B[i]);
    }

    // Clean up: destroy CUDA events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    // Free device memory
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
