/*
Kernel launches do not return an error code directly. Add error checking for a kernel launch by calling cudaPeekAtLastError() immediately after the launch and cudaDeviceSynchronize() sometime later. Wrap both of these calls with your CHECK() macro.

Thinking: The program demonstrates how to perform error checking after a kernel launch. We define a CHECK macro that checks the result of a CUDA runtime function and prints an error message with cudaGetErrorString if an error occurs, then exits. The kernel launch is followed by CHECK(cudaPeekAtLastError()) to capture any launch errors. Later, we call CHECK(cudaDeviceSynchronize()) to ensure any runtime errors that occur during kernel execution are caught. The example uses a simple vector addition kernel and allocates device memory for demonstration. This structure ensures that both launch-time and execution-time errors are caught and reported.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define CHECK(call)                                                     \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

__global__ void vectorAdd(const float* A, const float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = A[idx] + B[idx];
}

int main() {
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK(cudaMalloc((void**)&d_A, size));
    CHECK(cudaMalloc((void**)&d_B, size));
    CHECK(cudaMalloc((void**)&d_C, size));

    // Copy host to device
    CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Check for launch errors
    CHECK(cudaPeekAtLastError());

    // Copy result back to host
    CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Synchronize and check for execution errors
    CHECK(cudaDeviceSynchronize());

    // Verify results
    bool success = true;
    for (int i = 0; i < N; i++) {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5) {
            fprintf(stderr, "Mismatch at index %d: GPU %f, CPU %f\n", i, h_C[i], expected);
            success = false;
            break;
        }
    }

    if (success) {
        printf("Result verification passed.\n");
    }

    // Free memory
    free(h_A);
    free(h_B);
    free(h_C);
    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_B));
    CHECK(cudaFree(d_C));

    return 0;
}
