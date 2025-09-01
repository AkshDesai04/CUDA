```cpp
/* 
Pass the struct by constant reference (`const MyData&`) to the kernel. This can be more efficient for large structs.

Thinking process:
- We want to demonstrate how passing a large struct by const reference to a CUDA kernel reduces data transfer overhead compared to passing by value.
- Define a `MyData` struct containing a sizable array (`float arr[256]`) and a size field.
- Allocate an instance of `MyData` on the host, fill it with test data.
- Copy the struct to device memory using `cudaMalloc` and `cudaMemcpy`.
- The kernel will receive the struct as a `const MyData&` argument, meaning the struct is not copied by value; only the pointer/reference is passed.
- Inside the kernel, we simply double each element in the array and write the result to an output buffer.
- After kernel execution, copy the output back to the host and print a few values to verify correctness.
- Include basic error checking to ensure robustness.
*/

#include <cstdio>
#include <cuda_runtime.h>

#define ARRAY_SIZE 256
#define BLOCK_SIZE 256

// Structure containing a large array
struct MyData {
    int size;
    float arr[ARRAY_SIZE];
};

// CUDA kernel that receives the struct by const reference
__global__ void processData(const MyData& data, float* output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < data.size) {
        // Simple operation: double the value
        output[idx] = data.arr[idx] * 2.0f;
    }
}

// Macro for error checking
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",           \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

int main() {
    // Host data initialization
    MyData hostData;
    hostData.size = ARRAY_SIZE;
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        hostData.arr[i] = static_cast<float>(i);
    }

    // Device memory allocation for the struct
    MyData* devData = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&devData, sizeof(MyData)));
    CUDA_CHECK(cudaMemcpy(devData, &hostData, sizeof(MyData), cudaMemcpyHostToDevice));

    // Device memory allocation for output
    float* devOut = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&devOut, hostData.size * sizeof(float)));

    // Launch kernel
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((hostData.size + BLOCK_SIZE - 1) / BLOCK_SIZE);
    processData<<<gridDim, blockDim>>>(*devData, devOut);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    float hostOut[ARRAY_SIZE];
    CUDA_CHECK(cudaMemcpy(hostOut, devOut, hostData.size * sizeof(float), cudaMemcpyDeviceToHost));

    // Print first 10 results to verify
    printf("First 10 results after doubling:\n");
    for (int i = 0; i < 10; ++i) {
        printf("hostOut[%d] = %f\n", i, hostOut[i]);
    }

    // Clean up
    CUDA_CHECK(cudaFree(devData));
    CUDA_CHECK(cudaFree(devOut));

    return 0;
}
```