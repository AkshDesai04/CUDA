```cpp
/* 
Now do the reverse: allocate host memory with `malloc`, pass the host pointer to a kernel, and have the kernel try to dereference it. What happens? (This can cause an error or unspecified behavior).

Thinking:
The exercise requires us to allocate memory on the host using the standard C `malloc` function. 
The pointer returned by `malloc` points to memory in the CPU address space, not in the GPU address space. 
CUDA kernels execute on the GPU and have no direct visibility into CPU memory unless that memory
has been made accessible to the device via CUDA mechanisms such as unified memory (`cudaMallocManaged`),
pinned host memory (`cudaHostAlloc` / `cudaMallocHost`), or explicit memory copies (`cudaMemcpy`).

When we pass a host pointer obtained by `malloc` directly into a kernel as a kernel argument,
the GPU will interpret that pointer value as a device address. It will attempt to read or write
to that address as if it were a location in device memory. Because the address does not belong
to the device's address space, this is an invalid memory access. The behavior is undefined:
the kernel may simply read garbage values, write to an invalid location, or the CUDA runtime may
detect the invalid pointer and return an error status.

To observe the effect, we will:

1. Allocate an array on the host using `malloc`.
2. Pass the raw pointer to a CUDA kernel that attempts to write to the first element.
3. Launch the kernel and check for errors using `cudaGetLastError` and `cudaDeviceSynchronize`.
4. Print out any error messages.

We expect the program to either produce a CUDA error such as "invalid device pointer" or cause a
runtime crash, demonstrating that host-allocated memory cannot be dereferenced directly from a
device kernel without proper memory management.

This code is written in C++ but can be compiled with `nvcc` as a .cu file. It is self-contained
and includes all necessary error checking to illustrate the point.

*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Simple kernel that attempts to write to the first element of an array.
__global__ void tryDereference(int *hostPtr, int val)
{
    // Attempt to write to the host memory as if it were device memory.
    // This should cause an error or undefined behavior.
    hostPtr[0] = val;
}

int main()
{
    const int N = 10;
    const int bytes = N * sizeof(int);

    // Allocate host memory with malloc (unmanaged memory).
    int *hostArray = (int *)malloc(bytes);
    if (!hostArray)
    {
        fprintf(stderr, "Failed to allocate host memory with malloc\n");
        return EXIT_FAILURE;
    }

    // Initialize host array (optional).
    for (int i = 0; i < N; ++i)
    {
        hostArray[i] = i;
    }

    // Launch kernel with host pointer.
    tryDereference<<<1, 1>>>(hostArray, 42);

    // Check for kernel launch errors.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
    }

    // Synchronize to force the kernel to complete and capture any runtime errors.
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(err));
    }
    else
    {
        printf("Kernel executed successfully (unexpected).\n");
        printf("hostArray[0] = %d\n", hostArray[0]);
    }

    // Clean up host memory.
    free(hostArray);

    return 0;
}
```