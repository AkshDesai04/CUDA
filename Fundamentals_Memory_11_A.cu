/*
Aim: Explain in a comment the properties of constant memory (cached, read-only for kernel, low latency when all threads in a warp access the same address).

This program demonstrates how to use CUDA constant memory. Constant memory is a small, read‑only memory space on the device that is cached by the GPU’s constant cache. The key properties are:

* **Cached** – Each Streaming Multiprocessor (SM) has a constant cache that serves accesses to constant memory. This cache is typically 64 bytes in size (though the exact size can vary by architecture). Because the data is cached, repeated accesses by threads incur very low latency.

* **Read‑only for kernels** – Once data is copied into constant memory (using cudaMemcpyToSymbol or the `__constant__` qualifier), kernels can only read from it. The compiler can perform various optimizations knowing the memory will not be written to by the device.

* **Low latency when all threads in a warp access the same address** – If every thread in a warp reads the same address in constant memory, the constant cache can service the entire warp with a single read transaction. This is why constant memory is ideal for small lookup tables or parameters that are broadcast to many threads.

The program below defines a constant array, copies data from the host to this constant memory, launches a kernel that reads the constants, and writes results to a global output array. It then copies the results back to the host and prints them, illustrating the use of constant memory and its properties.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Size of the array we will store in constant memory
#define ARRAY_SIZE 10

// Constant memory declaration
__constant__ int d_constArray[ARRAY_SIZE];

// Kernel that reads from constant memory and writes to global memory
__global__ void useConstant(int *out)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < ARRAY_SIZE)
    {
        // Read the same constant value for all threads in the warp if idx % warpSize == 0
        // Here we simply read the element at the same index to demonstrate usage
        out[idx] = d_constArray[idx] * 2;
    }
}

int main()
{
    // Host array to be copied to constant memory
    int h_array[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; ++i)
        h_array[i] = i + 1;  // Fill with values 1..10

    // Copy to constant memory
    cudaMemcpyToSymbol(d_constArray, h_array, ARRAY_SIZE * sizeof(int));

    // Allocate output array on device
    int *d_out;
    cudaMalloc((void **)&d_out, ARRAY_SIZE * sizeof(int));

    // Launch kernel: one thread per element
    int threadsPerBlock = 32;
    int blocks = (ARRAY_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    useConstant<<<blocks, threadsPerBlock>>>(d_out);

    // Copy results back to host
    int h_out[ARRAY_SIZE];
    cudaMemcpy(h_out, d_out, ARRAY_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    // Print results
    printf("Input values in constant memory:\n");
    for (int i = 0; i < ARRAY_SIZE; ++i)
        printf("%d ", h_array[i]);
    printf("\n");

    printf("Kernel output (value * 2):\n");
    for (int i = 0; i < ARRAY_SIZE; ++i)
        printf("%d ", h_out[i]);
    printf("\n");

    // Clean up
    cudaFree(d_out);

    return 0;
}
