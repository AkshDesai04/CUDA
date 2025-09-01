```cuda
/* 
Aim: Compare __device__ memory to __constant__ memory. What are the key differences? (Writable by kernel, not cached in the same way).

Thinking:
1. __device__ memory is ordinary global memory that is accessible and writable by device code, and can be read/written from the host via cudaMemcpy. It is cached in the regular L2/global memory cache hierarchy.
2. __constant__ memory is a separate address space that is read‑only from device code, cached in a dedicated constant cache (broadcast cache). The host can write to it using cudaMemcpyToSymbol / cudaMemcpyFromSymbol.
3. In this example we:
   - declare a __device__ int devVar and a __constant__ int constVal.
   - the host initializes devVar to 0 and constVal to a fixed value (e.g., 42).
   - a kernel increments devVar by 1 and reads constVal to verify it is unchanged.
   - the kernel cannot write to constVal (trying to do so would be a compilation error, so we comment it out).
   - After kernel execution we copy devVar back to host and print it.
   - This demonstrates that __device__ memory is writable by kernels and accessed through the regular memory hierarchy, whereas __constant__ memory is read‑only for kernels and uses a separate cache.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Declare a __device__ variable (global memory, writable by kernels)
__device__ int devVar;

// Declare a __constant__ variable (read‑only for kernels, cached in constant cache)
__constant__ int constVal = 42;  // initialized at compile time

// Kernel that increments devVar and reads constVal
__global__ void kernel()
{
    // Increment devVar atomically
    atomicAdd(&devVar, 1);

    // Read constVal (should be 42)
    int val = constVal;

    // For demonstration, we could store the read value somewhere,
    // but here we just ensure it compiles and is read correctly.
    // The following write to constVal is illegal and would cause a compile error:
    // constVal = 100; // <--- Not allowed, comment out
}

int main()
{
    // Set devVar to 0 from host
    int h_devVar = 0;
    cudaMemcpyToSymbol(devVar, &h_devVar, sizeof(int));

    // Optional: copy a new value to constVal from host
    int h_constVal = 99;  // change at runtime
    cudaMemcpyToSymbol(constVal, &h_constVal, sizeof(int));

    // Launch kernel
    kernel<<<1, 1>>>();
    cudaDeviceSynchronize();

    // Copy back devVar to host
    cudaMemcpyFromSymbol(&h_devVar, devVar, sizeof(int));

    // Print results
    printf("devVar after kernel: %d (expected 1)\n", h_devVar);
    printf("constVal after kernel: %d (expected 99)\n", h_constVal);

    return 0;
}
```