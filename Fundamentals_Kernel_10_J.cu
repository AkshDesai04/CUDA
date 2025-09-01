/*
 * Pass the struct by constant reference (`const MyData&`) to the kernel. This can be more efficient for large structs.
 *
 * Thinking:
 * - In CUDA, kernel parameters must be in device-accessible memory. To pass a struct by constant reference, we allocate the struct on the device and then pass a reference to it when launching the kernel.
 * - We define a struct `MyData` that contains a large array (e.g., 1024 floats) and some scalar values to illustrate a large structure.
 * - We allocate a host instance, initialize it, allocate a device instance, copy data over, and then launch a kernel that takes a `const MyData&` parameter. Inside the kernel we perform some simple computation (e.g., summing the array) and store the result in another device variable.
 * - We use `printf` from device to demonstrate access to the struct contents.
 * - After kernel completion we copy the result back to host and print it.
 * - The code includes error checking helper to make debugging easier.
 * - Using a reference (`const MyData&`) instead of passing the whole struct by value avoids copying the entire struct from host to device for each kernel launch, which can be more efficient when the struct is large.
 * - The code compiles with `nvcc` and runs on a device that supports printf.
 */

#include <cstdio>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                               \
    do {                                                               \
        cudaError_t err = call;                                        \
        if (err != cudaSuccess) {                                      \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",          \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

// Define a large struct
struct MyData {
    float values[1024];   // Large array
    int   count;
    float factor;
};

// Device kernel that receives the struct by const reference
__global__ void processDataKernel(const MyData& data, float* result) {
    // Compute the sum of the array and multiply by factor
    float sum = 0.0f;
    for (int i = 0; i < data.count; ++i) {
        sum += data.values[i];
    }
    sum *= data.factor;

    // Write result to global memory
    *result = sum;

    // Optionally print some info from device
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("Kernel received MyData with count=%d, factor=%f\n", data.count, data.factor);
        printf("Computed sum: %f\n", sum);
    }
}

int main() {
    // Host side initialization
    MyData h_data;
    for (int i = 0; i < 1024; ++i) {
        h_data.values[i] = static_cast<float>(i) * 0.5f;
    }
    h_data.count = 1024;
    h_data.factor = 2.0f;

    // Allocate device memory for the struct
    MyData* d_data = nullptr;
    CHECK_CUDA(cudaMalloc(&d_data, sizeof(MyData)));

    // Copy struct from host to device
    CHECK_CUDA(cudaMemcpy(d_data, &h_data, sizeof(MyData), cudaMemcpyHostToDevice));

    // Allocate device memory for result
    float* d_result = nullptr;
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(float)));

    // Launch kernel with const reference to the struct
    processDataKernel<<<1, 1>>>(*d_data, d_result);
    CHECK_CUDA(cudaGetLastError());

    // Wait for GPU to finish
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back to host
    float h_result = 0.0f;
    CHECK_CUDA(cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));

    // Print result on host
    printf("Result from device: %f\n", h_result);

    // Clean up
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_result));

    return 0;
}
