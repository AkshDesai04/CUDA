/*
Aim: Time your SAXPY implementation for a large vector.

We will implement a CUDA kernel that computes C[i] = a*X[i] + Y[i] for each element of large vectors X, Y, and store in C.
We will measure the execution time of the kernel (excluding data transfer) using cudaEventRecord and cudaEventElapsedTime.
We'll allocate memory on host and device, initialize the host arrays with sample values, copy to device, launch kernel, copy result back if needed, and then clean up.
We also include error checking macro for CUDA calls.
We will use a large vector size (e.g., 1<<26 elements) to stress the GPU and get meaningful timing.
*/

#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",        \
                    __FILE__, __LINE__, cudaGetErrorString(err));               \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// SAXPY kernel: C[i] = a * X[i] + Y[i]
__global__ void saxpy_kernel(float *C, const float *X, const float *Y, float a, size_t N) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = a * X[idx] + Y[idx];
    }
}

int main() {
    const size_t N = 1 << 26; // 67,108,864 elements (~256 MB per array of floats)
    const float a = 2.5f;

    size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_X = (float*)malloc(bytes);
    float *h_Y = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    // Initialize host arrays
    for (size_t i = 0; i < N; ++i) {
        h_X[i] = static_cast<float>(i);
        h_Y[i] = static_cast<float>(i * 2);
    }

    // Allocate device memory
    float *d_X = nullptr;
    float *d_Y = nullptr;
    float *d_C = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_X, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_Y, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));

    // Copy input data from host to device
    CUDA_CHECK(cudaMemcpy(d_X, h_X, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Y, h_Y, bytes, cudaMemcpyHostToDevice));

    // Configure kernel launch parameters
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Record start event
    CUDA_CHECK(cudaEventRecord(start, 0));

    // Launch SAXPY kernel
    saxpy_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_X, d_Y, a, N);

    // Record stop event
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Calculate elapsed time
    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // Optionally copy result back to host
    // CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    std::cout << "SAXPY kernel execution time: " << milliseconds << " ms" << std::endl;

    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_Y));
    CUDA_CHECK(cudaFree(d_C));
    free(h_X);
    free(h_Y);
    free(h_C);

    return 0;
}
