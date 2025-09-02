```cpp
/* 
Use `cudaPeekAtLastError()` immediately after the kernel launch to get the error code without waiting for a synchronization.

Thinking Process:
1. Goal is to demonstrate how `cudaPeekAtLastError()` captures errors that occur at launch time, such as invalid configuration, without requiring an explicit device synchronization (`cudaDeviceSynchronize()`).
2. The simplest runtime error that triggers immediately upon kernel launch is an invalid launch configuration. CUDA limits the number of threads per block (e.g., 1024 on most devices). Exceeding this limit causes a launch failure.
3. I will write a trivial kernel that does nothing but will be launched with a thread count exceeding the device limit. This guarantees a launch error.
4. After the kernel launch I will call `cudaPeekAtLastError()` to retrieve the error code. If an error occurred, I will print it using `cudaGetErrorString()`. No synchronization call is needed.
5. To make the example complete, I also allocate some device memory, though it will not be used due to the launch failure. This shows a realistic scenario where device memory is involved.
6. The code is minimal, self‑contained, and can be compiled with `nvcc` to produce a `.cu` executable.

*/

#include <cstdio>
#include <cuda_runtime.h>

__global__ void dummyKernel(int *data)
{
    // Simple kernel that writes to the provided pointer
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    data[idx] = idx;
}

int main()
{
    const int N = 10;
    int *d_data;
    cudaError_t err;

    // Allocate device memory
    err = cudaMalloc((void**)&d_data, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Intentionally launch with too many threads per block to trigger a launch error
    dim3 blockDim(2048);   // Exceeds typical maximum (1024)
    dim3 gridDim(1);

    // Launch kernel
    dummyKernel<<<gridDim, blockDim>>>(d_data);

    // Immediately check for launch error without synchronization
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
    } else {
        // If no error, we would normally synchronize to catch other errors
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "Kernel execution error: %s\n", cudaGetErrorString(err));
        } else {
            printf("Kernel executed successfully.\n");
        }
    }

    // Clean up
    cudaFree(d_data);

    return 0;
}
```