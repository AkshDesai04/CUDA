/*
Implement an in-place version: `data[i] += scalar_value`.

Thinking:
1. The task is to write a CUDA program that performs an in-place addition of a scalar value to each element of an array. 
2. The program will allocate an array on the host, copy it to the device, launch a kernel that adds the scalar to each element in place, copy the data back, and then display the results. 
3. To keep the example simple and self‑contained, the array will be small (e.g., 10 elements) and will be initialized with known values so that the output is easy to verify. 
4. The scalar value will be a compile‑time constant, but it could be passed as a parameter if desired. 
5. We will use standard CUDA error checking macros for robustness. 
6. The code will be fully contained in a single .cu file, with comments to explain each part. 
7. The aim of the program is explicitly stated in a multiline comment as required by the instruction.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// CUDA error checking macro
#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        exit(EXIT_FAILURE); \
    }

// Kernel that adds a scalar value to each element of the array in-place
__global__ void add_scalar_inplace(float* data, float scalar, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += scalar;
    }
}

int main() {
    const int n = 10;                      // Number of elements
    const float scalar_value = 5.0f;       // Scalar to add

    // Host memory allocation
    float h_data[n];
    for (int i = 0; i < n; ++i) {
        h_data[i] = static_cast<float>(i); // Initialize with 0,1,2,...
    }

    // Device memory allocation
    float* d_data = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_data, n * sizeof(float)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    add_scalar_inplace<<<blocksPerGrid, threadsPerBlock>>>(d_data, scalar_value, n);
    CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(d_data));

    // Print results
    printf("Result after in-place addition of %.2f:\n", scalar_value);
    for (int i = 0; i < n; ++i) {
        printf("h_data[%d] = %.2f\n", i, h_data[i]);
    }

    return 0;
}
