/*
Pass a `nullptr` to the kernel and have the kernel try to read or write from it. Analyze the `cuda-memcheck` report for this null pointer dereference.

Thinking:
- The purpose of the program is to intentionally pass a null pointer from the host to a CUDA kernel and then dereference it within the kernel. This should trigger a memory access violation that cuda-memcheck will catch and report.
- The kernel accepts an `int *` argument, which will be a device pointer. On the host side we set the pointer to `nullptr` (i.e., 0) and pass it directly to the kernel launch. Because the pointer is not a valid device memory address, any attempt to read from or write to it inside the kernel will result in an illegal memory access.
- Inside the kernel we will perform a simple write `ptr[idx] = idx;` and optionally a read `int val = ptr[idx];`. This access will cause the GPU to attempt to read/write address 0, which is not a valid device memory region.
- After launching the kernel we check for launch errors with `cudaGetLastError()` and after synchronization with `cudaDeviceSynchronize()` to catch any runtime errors that occur during execution.
- To analyze the failure, one would compile the code with `nvcc` and run it under `cuda-memcheck`:
    ```
    nvcc -o null_deref null_deref.cu
    cuda-memcheck ./null_deref
    ```
  The memcheck output will report an "Invalid device pointer" or "Invalid address accessed" error at the line where the dereference occurs, indicating the kernel attempted to access memory address 0.
- The code is deliberately minimal and self‑contained; no device memory is allocated because we intentionally want the null pointer dereference to happen.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void nullDereferenceKernel(int *ptr, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        // Attempt to write to the null pointer
        ptr[idx] = idx;

        // Optionally, attempt to read from the null pointer
        int val = ptr[idx];
        (void)val;  // Suppress unused variable warning
    }
}

int main(void)
{
    const int N = 10;

    // Intentionally pass a null pointer (device memory not allocated)
    int *devPtr = nullptr;

    // Launch kernel with the null pointer
    nullDereferenceKernel<<<1, N>>>(devPtr, N);

    // Check for errors in kernel launch
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
    }

    // Synchronize to catch execution errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        printf("Kernel execution error: %s\n", cudaGetErrorString(err));
    }

    // Clean up (nothing to free since devPtr is null)
    cudaDeviceReset();

    return 0;
}
