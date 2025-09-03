/*
Use an array in constant memory, e.g., `__constant__ float coefficients[16];`. Copy 16 floats from the host and use them in a kernel.

Thinking process:
1. Define a constant memory array of 16 floats on the device: `__constant__ float coefficients[16];`
2. On the host, create a matching array of 16 floats, initialize it with some values.
3. Copy the host array into the device constant memory using `cudaMemcpyToSymbol`.
4. Write a simple kernel that reads each coefficient from constant memory, performs a basic operation (e.g., multiply by 2), and writes the result to an output array in global memory.
5. Allocate an output array on the device, launch the kernel with one thread per coefficient, then copy the output back to the host.
6. Print the results to verify that constant memory was accessed correctly.
7. Add basic CUDA error checking to help debug.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Macro for error checking
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",               \
                    __FUNCTION__, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Constant memory array on device
__constant__ float coefficients[16];

// Kernel that uses constant memory
__global__ void useConstants(float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = coefficients[idx] * 2.0f; // simple operation
    }
}

int main() {
    const int N = 16;
    size_t size = N * sizeof(float);

    // Host arrays
    float h_coeffs[N];
    float h_result[N];

    // Initialize host coefficients
    for (int i = 0; i < N; ++i) {
        h_coeffs[i] = static_cast<float>(i + 1); // 1.0, 2.0, ..., 16.0
    }

    // Copy coefficients to constant memory
    CUDA_CHECK(cudaMemcpyToSymbol(coefficients, h_coeffs, size, 0, cudaMemcpyHostToDevice));

    // Allocate device output array
    float* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, size));

    // Launch kernel: one block with N threads
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    useConstants<<<blocks, threadsPerBlock>>>(d_out, N);
    CUDA_CHECK(cudaGetLastError()); // Check for launch errors

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_result, d_out, size, cudaMemcpyDeviceToHost));

    // Print results
    printf("Result (coefficients multiplied by 2):\n");
    for (int i = 0; i < N; ++i) {
        printf("coeff[%d] = %f -> out[%d] = %f\n", i, h_coeffs[i], i, h_result[i]);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}
