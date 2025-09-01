/*
Aim: Create a `__device__` function that takes a pointer to the struct (`MyData*`) and performs an operation on it.

Thinking:
- Define a simple struct `MyData` with two float members to keep the example lightweight.
- The `__device__` function `modifyData` will accept a pointer to `MyData` and perform a basic arithmetic operation, e.g., adding the second member to the first.
- A CUDA kernel `modifyKernel` will iterate over an array of `MyData` and call `modifyData` for each element.
- The host code will:
  * Allocate an array of `MyData` on the host.
  * Allocate corresponding memory on the device.
  * Copy the host array to the device.
  * Launch the kernel with an appropriate grid/block configuration.
  * Copy the modified array back to the host.
  * Print the results to verify the operation.
- Simple error checking macros are added to make debugging easier.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Error checking macro
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

// Define the struct
struct MyData {
    float a;
    float b;
};

// Device function that operates on MyData*
__device__ void modifyData(MyData* d) {
    // Example operation: add b to a
    d->a += d->b;
}

// Kernel that calls the device function for each element
__global__ void modifyKernel(MyData* d, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        modifyData(&d[idx]);
    }
}

int main(void) {
    const int N = 10;
    size_t size = N * sizeof(MyData);

    // Host allocation and initialization
    MyData h_data[N];
    for (int i = 0; i < N; ++i) {
        h_data[i].a = i * 1.0f;
        h_data[i].b = (N - i) * 0.5f;
    }

    // Device allocation
    MyData* d_data;
    CHECK_CUDA(cudaMalloc((void**)&d_data, size));

    // Copy from host to device
    CHECK_CUDA(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    // Kernel launch configuration
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    modifyKernel<<<blocks, threadsPerBlock>>>(d_data, N);
    CHECK_CUDA(cudaGetLastError());

    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));

    // Print results
    printf("Modified data:\n");
    for (int i = 0; i < N; ++i) {
        printf("Index %d: a = %f, b = %f\n", i, h_data[i].a, h_data[i].b);
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_data));

    return 0;
}
