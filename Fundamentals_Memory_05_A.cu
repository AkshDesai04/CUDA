```cuda
/*
Aim of the program: Modify the program to allocate and free the memory in a loop 100 times.
Thinking:
The task requires a CUDA program that demonstrates repeated allocation and deallocation
of GPU memory to verify that the device can handle such operations without leaks or
resource exhaustion. The program will:
1. Allocate a host array of floats (size N) and initialize it.
2. Inside a loop that iterates 100 times:
   a. Allocate device memory for the array.
   b. Copy the host array to the device.
   c. Launch a simple kernel that performs a trivial operation (e.g., scaling each element).
   d. Copy the processed data back to the host (optional, just to show a complete round trip).
   e. Free the device memory.
3. Include error checking after each CUDA API call to catch and report failures.
4. Use a small kernel for demonstration; the focus is on allocation/deallocation rather than computation.
5. Provide minimal output to indicate progress (e.g., iteration number) and final success message.
6. Ensure that the code compiles as a standalone .cu file with nvcc and runs on a CUDA-capable device.
*/

#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>

// Error checking macro
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// Simple kernel that multiplies each element by 2.0f
__global__ void scaleKernel(float* d_arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_arr[idx] *= 2.0f;
    }
}

int main() {
    const int N = 1 << 20;           // 1M elements
    const int ITERATIONS = 100;      // Number of allocation/deallocation cycles

    // Allocate and initialize host array
    float* h_arr = new float[N];
    for (int i = 0; i < N; ++i) {
        h_arr[i] = static_cast<float>(i);
    }

    // Define CUDA grid/block dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        float* d_arr = nullptr;

        // Allocate device memory
        CHECK_CUDA(cudaMalloc((void**)&d_arr, N * sizeof(float)));

        // Copy data from host to device
        CHECK_CUDA(cudaMemcpy(d_arr, h_arr, N * sizeof(float), cudaMemcpyHostToDevice));

        // Launch kernel
        scaleKernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr, N);
        CHECK_CUDA(cudaGetLastError()); // Check for kernel launch errors

        // Optionally copy back to host (not strictly necessary for allocation test)
        CHECK_CUDA(cudaMemcpy(h_arr, d_arr, N * sizeof(float), cudaMemcpyDeviceToHost));

        // Free device memory
        CHECK_CUDA(cudaFree(d_arr));

        // Print progress
        std::cout << "Iteration " << iter + 1 << " completed." << std::endl;
    }

    // Clean up host memory
    delete[] h_arr;

    std::cout << "All " << ITERATIONS << " allocation/deallocation cycles completed successfully." << std::endl;

    return 0;
}
```