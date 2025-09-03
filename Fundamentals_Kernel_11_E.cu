```cpp
/*
From the host, use `cudaMemset` to zero out the struct's memory on the device after allocating it.

Thinking:
1. Define a simple struct on the host and device side (same layout) to hold some example data.
2. Allocate memory for this struct on the device with cudaMalloc.
3. Use cudaMemset from the host to set all bytes of the allocated struct to zero.
4. Copy the struct back to the host with cudaMemcpy to verify that all fields are zeroed.
5. Provide minimal error checking for CUDA API calls.
6. The code is a self‑contained .cu file that compiles with nvcc and prints the values of the struct before and after the memset operation.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Simple struct with different types
struct Data {
    int   i;
    float f;
    double d;
};

// Utility macro for CUDA error checking
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                 \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));      \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

int main() {
    // Allocate host instance and initialize with non-zero values
    Data hostData{42, 3.14f, 2.718};
    printf("Host before memset: i=%d, f=%f, d=%f\n", hostData.i, hostData.f, hostData.d);

    // Device pointer
    Data* devPtr = nullptr;
    CUDA_CHECK(cudaMalloc(&devPtr, sizeof(Data)));

    // Copy initial data to device
    CUDA_CHECK(cudaMemcpy(devPtr, &hostData, sizeof(Data), cudaMemcpyHostToDevice));

    // Use cudaMemset to zero out the struct on the device
    CUDA_CHECK(cudaMemset(devPtr, 0, sizeof(Data)));

    // Copy back to host to verify
    CUDA_CHECK(cudaMemcpy(&hostData, devPtr, sizeof(Data), cudaMemcpyDeviceToHost));

    printf("Host after memset: i=%d, f=%f, d=%f\n", hostData.i, hostData.f, hostData.d);

    // Clean up
    CUDA_CHECK(cudaFree(devPtr));

    return 0;
}
```