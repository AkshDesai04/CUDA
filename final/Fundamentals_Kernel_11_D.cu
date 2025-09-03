/*
Create a `__device__` function that takes a pointer to the struct (`MyData*`) and performs an operation on it.

Thinking:
- Define a simple struct `MyData` with a couple of members.
- Write a `__device__` function that takes a `MyData*` and modifies its members.
- Provide a kernel that calls this device function for demonstration.
- Add a minimal host `main` to allocate, initialize, launch kernel, copy back, and print results.
- Keep the code self‑contained and minimal but functional for compiling with nvcc.
*/

#include <cstdio>
#include <cuda_runtime.h>

struct MyData {
    int   a;
    float b;
};

// Device function that modifies the data pointed to by ptr
__device__ void processMyData(MyData* ptr) {
    // Example operation: increment 'a' and double 'b'
    ptr->a += 1;
    ptr->b *= 2.0f;
}

// Kernel that calls the device function on each element
__global__ void kernel(MyData* d_data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        processMyData(&d_data[idx]);
    }
}

int main() {
    const int n = 4;
    MyData h_data[n];

    // Initialize host data
    for (int i = 0; i < n; ++i) {
        h_data[i].a = i;
        h_data[i].b = static_cast<float>(i) * 1.5f;
    }

    // Allocate device memory
    MyData* d_data = nullptr;
    cudaMalloc(&d_data, n * sizeof(MyData));
    cudaMemcpy(d_data, h_data, n * sizeof(MyData), cudaMemcpyHostToDevice);

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    kernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, n);
    cudaDeviceSynchronize();

    // Copy back results
    cudaMemcpy(h_data, d_data, n * sizeof(MyData), cudaMemcpyDeviceToHost);

    // Print results
    for (int i = 0; i < n; ++i) {
        printf("Element %d: a = %d, b = %f\n", i, h_data[i].a, h_data[i].b);
    }

    // Clean up
    cudaFree(d_data);
    return 0;
}