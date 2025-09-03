/*
Time the multiplication kernel.

The goal of this program is to demonstrate how to measure the execution time of a CUDA kernel that performs element‑wise multiplication of two vectors.  
We choose a simple kernel because it isolates the timing of GPU execution without extra overhead such as memory allocation or data transfer.  

Design decisions:
1. Use `float` arrays for straightforward arithmetic.
2. Use a one‑dimensional grid and block configuration that covers all elements.
3. Allocate host memory with `malloc`, fill it with deterministic values, then copy to the device.
4. Use CUDA events (`cudaEvent_t`) to time only the kernel launch, not the data transfer.
5. Add a simple error‑checking macro `checkCudaErrors` to simplify CUDA call handling.
6. Print the elapsed time in milliseconds.
7. Clean up device memory and events before exit.

The program is self‑contained and can be compiled with `nvcc`. It will print the time taken by the kernel in milliseconds for the chosen problem size.
*/

#include <iostream>
#include <cuda_runtime.h>

// Simple macro to check CUDA errors
#define checkCudaErrors(call)                                               \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            std::cerr << "CUDA error in file '" << __FILE__                  \
                      << "' in line " << __LINE__ << ": "                   \
                      << cudaGetErrorString(err) << std::endl;              \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Kernel performing element-wise multiplication of two vectors
__global__ void vectorMul(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] * B[idx];
    }
}

int main() {
    // Problem size
    const int N = 1 << 20; // 1M elements
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i) * 0.001f;
        h_B[i] = static_cast<float>(i) * 0.002f;
    }

    // Allocate device memory
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    checkCudaErrors(cudaMalloc((void **)&d_A, bytes));
    checkCudaErrors(cudaMalloc((void **)&d_B, bytes));
    checkCudaErrors(cudaMalloc((void **)&d_C, bytes));

    // Copy data from host to device
    checkCudaErrors(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Timing events
    cudaEvent_t start, stop;
    checkCudaErrors(cudaEventCreate(&start));
    checkCudaErrors(cudaEventCreate(&stop));

    // Define execution configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Record start event
    checkCudaErrors(cudaEventRecord(start, 0));

    // Launch kernel
    vectorMul<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    checkCudaErrors(cudaGetLastError());

    // Record stop event
    checkCudaErrors(cudaEventRecord(stop, 0));
    checkCudaErrors(cudaEventSynchronize(stop));

    // Compute elapsed time
    float milliseconds = 0;
    checkCudaErrors(cudaEventElapsedTime(&milliseconds, start, stop));

    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;

    // Optional: copy result back to host (not timed)
    checkCudaErrors(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Clean up
    checkCudaErrors(cudaEventDestroy(start));
    checkCudaErrors(cudaEventDestroy(stop));
    checkCudaErrors(cudaFree(d_A));
    checkCudaErrors(cudaFree(d_B));
    checkCudaErrors(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
