```cuda
/* 
Aim of the program:
Create a `__device__` function that takes a pointer to the struct (`MyData*`) and performs an operation on it.

Thinking:
1. Define a simple struct `MyData` with a couple of fields to illustrate manipulation.
2. Write a `__device__` function `modifyData` that accepts a pointer to `MyData` and changes its contents (e.g., increment an integer and scale a float).
3. Provide a CUDA kernel that calls `modifyData` for each element in an array of `MyData`.
4. In the host code, allocate an array of `MyData` on the device, initialize it on the host, copy it over, launch the kernel, copy the results back, and print them to confirm the operation.
5. The program is self‑contained in a single `.cu` file and uses only standard CUDA runtime APIs.
*/

#include <cstdio>
#include <cuda_runtime.h>

// --------------------------------------------------
// Definition of the data structure
// --------------------------------------------------
struct MyData {
    int   id;
    float value;
};

// --------------------------------------------------
// Device function that operates on MyData pointer
// --------------------------------------------------
__device__ void modifyData(MyData* d) {
    // Example operation: increment id, double the value
    d->id += 1;
    d->value *= 2.0f;
}

// --------------------------------------------------
// Kernel that calls the device function on each element
// --------------------------------------------------
__global__ void kernel_modify(MyData* d, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        modifyData(&d[idx]);   // call device function
    }
}

// --------------------------------------------------
// Helper macro for error checking
// --------------------------------------------------
#define CUDA_CHECK(call)                                         \
    do {                                                         \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess) {                                \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",    \
                    #call, __FILE__, __LINE__,                  \
                    cudaGetErrorString(err));                    \
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

// --------------------------------------------------
// Main host routine
// --------------------------------------------------
int main(void) {
    const int N = 10;
    MyData hData[N];

    // Initialize host data
    for (int i = 0; i < N; ++i) {
        hData[i].id    = i;
        hData[i].value = static_cast<float>(i) + 0.5f;
    }

    // Allocate device memory
    MyData *dData = nullptr;
    CUDA_CHECK(cudaMalloc(&dData, N * sizeof(MyData)));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(dData, hData, N * sizeof(MyData), cudaMemcpyHostToDevice));

    // Launch kernel: 1 block of N threads (or adjust as needed)
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    kernel_modify<<<blocks, threadsPerBlock>>>(dData, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(hData, dData, N * sizeof(MyData), cudaMemcpyDeviceToHost));

    // Free device memory
    CUDA_CHECK(cudaFree(dData));

    // Print results
    printf("Modified data:\n");
    for (int i = 0; i < N; ++i) {
        printf("  element %2d: id = %3d, value = %7.2f\n",
               i, hData[i].id, hData[i].value);
    }

    return 0;
}
```