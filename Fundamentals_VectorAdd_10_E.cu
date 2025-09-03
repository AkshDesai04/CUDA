/*
Aim: For the GPU timing, use `cudaEvent`s around the kernel launch only.

Thinking: 
- The task is to demonstrate how to measure GPU execution time using CUDA events.  
- I will write a simple vector addition kernel that runs on the GPU.  
- Host memory will be allocated for two input arrays and one output array.  
- Device memory will be allocated for the same arrays.  
- After copying the inputs to the device, I will create two cudaEvents (`start` and `stop`).  
- I will record the `start` event *just before* the kernel launch and record the `stop` event *immediately after* the kernel launch.  
- Then I will synchronize on the `stop` event and use `cudaEventElapsedTime` to obtain the elapsed time in milliseconds.  
- Finally, the result will be copied back to host, validated, and the timing printed.  
- The code will include a simple error‑checking macro for clarity.  
- All the code is written in a single `.cu` file, with no external dependencies beyond the CUDA runtime.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Macro for checking CUDA errors following a CUDA API call
#define CHECK_CUDA(call)                                            \
    {                                                               \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in file '%s' in line %d : %s.\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    }

// Simple vector addition kernel
__global__ void vectorAdd(const float *A, const float *B, float *C, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void) {
    const size_t N = 1 << 20; // 1M elements
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    // Initialize host vectors
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    // Allocate device memory
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    CHECK_CUDA(cudaMalloc((void**)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_B, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_C, bytes));

    // Copy host input data to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Configure kernel launch parameters
    const int blockSize = 256;
    const int gridSize = (int)((N + blockSize - 1) / blockSize);

    // Record the start event
    CHECK_CUDA(cudaEventRecord(start, 0));

    // Launch kernel
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);

    // Record the stop event
    CHECK_CUDA(cudaEventRecord(stop, 0));

    // Wait for the stop event to complete
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Calculate elapsed time
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify results (simple check)
    bool success = true;
    for (size_t i = 0; i < N; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5) {
            fprintf(stderr, "Mismatch at index %zu: host %f, device %f, expected %f\n",
                    i, h_C[i], expected, expected);
            success = false;
            break;
        }
    }

    if (success) {
        printf("Vector addition successful. GPU kernel execution time: %.3f ms\n", milliseconds);
    } else {
        printf("Vector addition failed.\n");
    }

    // Clean up
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return EXIT_SUCCESS;
}
