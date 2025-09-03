```cuda
/*
Aim of the program:
Compare `__device__` memory to `__constant__` memory. What are the key differences? (Writable by kernel, not cached in the same way).

Thinking:
- `__device__` memory is global memory accessible by all threads; it is writable by kernels and has no special caching semantics beyond the normal L1/L2 cache hierarchy.
- `__constant__` memory is a read‑only memory region from the perspective of device code. It can only be written to from the host via `cudaMemcpyToSymbol` or during declaration. This memory is cached on the GPU and is especially efficient when all threads read the same value (broadcast).
- The program will demonstrate:
  1. A kernel that writes to a `__device__` array.
  2. An attempt to write to a `__constant__` array from a kernel (should fail to compile if uncommented, illustrating read‑only nature).
  3. Reading the `__constant__` array inside a kernel and printing the values.
  4. Writing to `__constant__` from the host and verifying the values on the device.
  5. A simple timing comparison is omitted for brevity, but comments explain how caching differences might affect performance.

The code below is a self‑contained example that can be compiled with `nvcc` and executed on a CUDA capable device.
*/

#include <cstdio>
#include <cuda_runtime.h>

#define N 10

// Device memory that is writable by kernels
__device__ int devArray[N];

// Constant memory that is read-only from device code
__constant__ int constArray[N] = {0};

// Kernel to write to device memory
__global__ void writeDeviceKernel(int value)
{
    int idx = threadIdx.x;
    if (idx < N)
    {
        devArray[idx] = value + idx;  // simple write
    }
}

// Kernel to read from constant memory and print values
__global__ void readConstantKernel()
{
    int idx = threadIdx.x;
    if (idx < N)
    {
        printf("Thread %d reading constArray[%d] = %d\n", idx, idx, constArray[idx]);
    }
}

// Kernel that attempts to write to constant memory (will not compile if uncommented)
// __global__ void writeConstantKernel()
// {
//     int idx = threadIdx.x;
//     if (idx < N)
//     {
//         constArray[idx] = idx; // Error: constArray is read-only for device code
//     }
// }

int main()
{
    // Host array to copy into constant memory
    int hostConst[N];
    for (int i = 0; i < N; ++i) hostConst[i] = i * 10;

    // Copy data from host to constant memory
    cudaMemcpyToSymbol(constArray, hostConst, N * sizeof(int), 0, cudaMemcpyHostToDevice);

    // Launch kernel to write to device memory
    writeDeviceKernel<<<1, N>>>(100);
    cudaDeviceSynchronize();

    // Launch kernel to read constant memory
    readConstantKernel<<<1, N>>>();
    cudaDeviceSynchronize();

    // Read back device memory to host to verify
    int hostDev[N];
    cudaMemcpyFromSymbol(hostDev, devArray, N * sizeof(int), 0, cudaMemcpyDeviceToHost);

    printf("\nValues in devArray after kernel write:\n");
    for (int i = 0; i < N; ++i)
    {
        printf("devArray[%d] = %d\n", i, hostDev[i]);
    }

    // Cleanup and exit
    cudaDeviceReset();
    return 0;
}
```