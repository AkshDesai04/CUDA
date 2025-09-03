/*
Create a C/C++ preprocessor macro `CHECK(call)` that wraps a CUDA API call. 
The macro should check the returned `cudaError_t` value. If it's not `cudaSuccess`, it should print the error code, the error string from `cudaGetErrorString`, the file name (`__FILE__`), and the line number (`__LINE__`), and then exit the program.

Thinking process:
1. Include necessary headers: <cuda_runtime.h> for CUDA types and functions, <stdio.h> for printing, <stdlib.h> for exit().
2. Define the macro as a do-while(0) block so it behaves like a single statement.
3. Execute the call, store result in a local variable of type cudaError_t.
4. Check if the result is not cudaSuccess. If so, print error details: file, line, error code, error string.
5. Call exit(EXIT_FAILURE) to terminate program.
6. Ensure macro is safe for expressions and side effects.
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(call)                                            \
    do {                                                       \
        cudaError_t err = (call);                              \
        if (err != cudaSuccess) {                              \
            fprintf(stderr, "CUDA error at %s:%d: %d %s\n",    \
                    __FILE__, __LINE__, err,                   \
                    cudaGetErrorString(err));                  \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    } while (0)

// Example kernel
__global__ void dummyKernel(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] = idx;
}

int main(void) {
    const int N = 1024;
    int *d_data;
    size_t size = N * sizeof(int);

    CHECK(cudaMalloc((void**)&d_data, size));

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    dummyKernel<<<grid, block>>>(d_data, N);
    CHECK(cudaGetLastError()); // Check kernel launch
    CHECK(cudaDeviceSynchronize()); // Check kernel execution

    CHECK(cudaFree(d_data));
    return 0;
}
