```cuda
/*
Aim of the program: Create a __device__ function that takes a pointer to the struct (MyData*) and performs an operation on it.

Thinking:
- We need a struct that the device function can modify. We'll define a simple struct `MyData` with an integer field `value` and possibly a float field for demonstration.
- The __device__ function will accept a pointer to `MyData` and perform a simple operation: increment the integer value by 1.0 and multiply the float field by 2.0. This demonstrates how a device function can dereference a pointer and modify the contents.
- To test the device function, we'll write a kernel that iterates over an array of `MyData` and calls the device function on each element.
- The host program will:
  1. Allocate and initialize an array of `MyData` on the host.
  2. Allocate memory on the device, copy the array over.
  3. Launch the kernel with enough threads to cover all elements.
  4. Copy the results back to the host and print them to verify the operation.
- Error checking is minimal but sufficient for demonstration.
- The entire code is self‑contained in a single .cu file with includes, struct definition, __device__ function, kernel, and main.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Simple data structure to be processed on the device
struct MyData {
    int   value;
    float factor;
};

// Device function that takes a pointer to MyData and performs an operation
__device__ void processMyData(MyData* data) {
    // Increment the integer value
    data->value += 1;
    // Double the factor
    data->factor *= 2.0f;
}

// Kernel that applies the device function to each element of an array
__global__ void kernelProcess(MyData* d_array, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        processMyData(&d_array[idx]);
    }
}

int main() {
    const int N = 5;
    MyData h_array[N];

    // Initialize host array
    for (int i = 0; i < N; ++i) {
        h_array[i].value = i;
        h_array[i].factor = 1.0f + i * 0.5f;
    }

    // Allocate device memory
    MyData* d_array;
    cudaMalloc((void**)&d_array, N * sizeof(MyData));

    // Copy data to device
    cudaMemcpy(d_array, h_array, N * sizeof(MyData), cudaMemcpyHostToDevice);

    // Launch kernel: one thread per element
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    kernelProcess<<<blocks, threadsPerBlock>>>(d_array, N);

    // Copy results back to host
    cudaMemcpy(h_array, d_array, N * sizeof(MyData), cudaMemcpyDeviceToHost);

    // Print results
    for (int i = 0; i < N; ++i) {
        printf("Element %d: value = %d, factor = %f\n",
               i, h_array[i].value, h_array[i].factor);
    }

    // Clean up
    cudaFree(d_array);
    return 0;
}
```