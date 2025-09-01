/*
Aim of the program: Create a header file (my_device_functions.cuh) and put your __device__ function in it. Include this header in your main .cu file.

Thinking: We need a simple CUDA program that demonstrates a __device__ function defined in a separate header. The header will declare a __device__ function, e.g., add. The .cu file will include the header and define a kernel that calls the device function. The main will launch kernel and copy results back to host. We must ensure that the header file is referenced correctly. Because the developer instruction says only output the .cu file, we will only provide the main .cu file and assume that the header exists separately. The header should be named my_device_functions.cuh. We will include it at the top. The program will be minimal but functional. We will use error checking macros. The program will allocate device array, copy input, launch kernel, copy output, and print results.
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include "my_device_functions.cuh" // Header containing __device__ function

// Kernel that uses the device function from the header
__global__ void computeKernel(const int* in, int* out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        // Use the __device__ function defined in the header
        out[idx] = add(in[idx], 1); // Example: add 1 to each element
    }
}

// Macro for error checking
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

int main(void) {
    const int N = 10;
    const int size = N * sizeof(int);
    int h_in[N], h_out[N];

    // Initialize input array
    for (int i = 0; i < N; ++i) {
        h_in[i] = i;
    }

    // Allocate device memory
    int *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_in, size));
    CUDA_CHECK(cudaMalloc((void**)&d_out, size));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    computeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));

    // Print results
    printf("Input : ");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_in[i]);
    }
    printf("\nOutput: ");
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_out[i]);
    }
    printf("\n");

    // Clean up
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    return 0;
}
