```cuda
/*
Aim: Add the `CHECK` macro around a call to `cudaGetLastError()`.

My thinking:
- The goal is to demonstrate the use of a `CHECK` macro that wraps a CUDA API call,
  specifically `cudaGetLastError()`.  
- The macro should capture the result of the call, compare it to `cudaSuccess`,
  and if an error occurs, print a helpful message including the file name,
  line number, and the CUDA error string, then exit the program.
- To illustrate usage, I will create a simple CUDA kernel that does nothing,
  launch it, and then immediately call `CHECK(cudaGetLastError())`.  
- I will also call `CHECK(cudaDeviceSynchronize())` to catch any errors that
  might arise during kernel execution.  
- The program will compile to a `.cu` file and can be built with `nvcc`.

The complete code follows:
*/

#include <stdio.h>
#include <stdlib.h>

// CHECK macro to wrap CUDA API calls
#define CHECK(call)                                                     \
    do {                                                                \
        cudaError_t err = (call);                                       \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));       \
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

// A trivial kernel that does nothing
__global__ void noopKernel(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx];
    }
}

int main(void) {
    const int N = 256;
    int *d_data = NULL;

    // Allocate device memory
    CHECK(cudaMalloc((void**)&d_data, N * sizeof(int)));

    // Launch kernel
    int threadsPerBlock = 64;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    noopKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);

    // Check for errors immediately after kernel launch
    CHECK(cudaGetLastError());

    // Wait for the kernel to finish
    CHECK(cudaDeviceSynchronize());

    // Free device memory
    CHECK(cudaFree(d_data));

    printf("Program completed successfully.\n");
    return 0;
}
```