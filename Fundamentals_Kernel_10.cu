```cpp
/*
Aim of the program:
Define a simple C struct on the host. Pass an instance of this struct by value to a kernel. The kernel should print one of the struct's members to verify it was received correctly.

Thinking and design:
1. Create a C struct named Simple with two members (int a and float b) to keep it small and simple.
2. Mark the struct definition with __host__ __device__ qualifiers so that it is visible in both host and device code. This is not strictly required for simple data types but makes the code explicit.
3. Define a kernel __global__ void printStruct(Simple s) that receives the struct by value. Inside the kernel, use device printf to output one of the members (s.a) to verify correct transfer.
4. In main(), instantiate a Simple object on the host, initialize its members, and call the kernel with a single thread block and single thread (<<<1,1>>>).
5. Add error checking for kernel launch and synchronization to ensure the program runs correctly.
6. Compile with NVCC targeting compute capability 2.0 or higher (default for most GPUs) to support device printf.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Define a simple C struct with host and device visibility
typedef struct {
    int a;
    float b;
} __host__ __device__ Simple;

// Kernel that receives the struct by value and prints one member
__global__ void printStruct(Simple s)
{
    // Use device printf (requires compute capability >= 2.0)
    printf("Device received struct member a = %d\n", s.a);
}

int main()
{
    // Create and initialize a Simple instance on the host
    Simple hostStruct;
    hostStruct.a = 42;
    hostStruct.b = 3.14f;

    // Launch kernel with one thread to pass the struct by value
    printStruct<<<1, 1>>>(hostStruct);

    // Check for launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Wait for device to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Device synchronization failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    return 0;
}
```