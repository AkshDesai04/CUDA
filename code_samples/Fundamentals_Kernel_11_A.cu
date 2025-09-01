/*
The kernel receives `MyData* d_data`. Modify it to read a value: `int x = d_data->a;` and then modify another value: `d_data->b = 2.0f * x;`.

Thinking:
1. Define a simple struct `MyData` with an integer `a` and a float `b`.
2. Write a CUDA kernel that takes a pointer to `MyData`, reads `a` into a local integer `x`, and then writes back to `b` the value `2.0f * x`.
3. In `main`, allocate a single instance of `MyData` on the host, initialize `a` and `b`.
4. Allocate memory on the device, copy the host struct to device memory.
5. Launch the kernel with one thread (since we only have one struct).
6. Copy the modified struct back to host, print the results, and clean up.
7. Include basic error checking for CUDA API calls.
*/

#include <cstdio>
#include <cuda_runtime.h>

struct MyData {
    int a;
    float b;
};

// Kernel that modifies the data as specified
__global__ void modifyKernel(MyData* d_data) {
    int x = d_data->a;
    d_data->b = 2.0f * x;
}

// Simple error checking macro
#define CUDA_CHECK(err) \
    do { \
        cudaError_t err__ = (err); \
        if (err__ != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

int main() {
    // Host data
    MyData h_data;
    h_data.a = 5;
    h_data.b = 0.0f;

    // Device pointer
    MyData* d_data = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_data, sizeof(MyData)));

    // Copy host data to device
    CUDA_CHECK(cudaMemcpy(d_data, &h_data, sizeof(MyData), cudaMemcpyHostToDevice));

    // Launch kernel: one block, one thread
    modifyKernel<<<1, 1>>>(d_data);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(&h_data, d_data, sizeof(MyData), cudaMemcpyDeviceToHost));

    // Print results
    printf("After kernel execution:\n");
    printf("a = %d\n", h_data.a);
    printf("b = %f\n", h_data.b);

    // Cleanup
    CUDA_CHECK(cudaFree(d_data));

    return 0;
}
