/*
Aim: What is the scope and lifetime of a __device__ variable? Explain in a comment. (Scope of the entire CUDA context, lifetime of the application).

Thinking and explanation:
1. A variable declared with the __device__ qualifier resides in the GPU device memory space.
2. The scope of a __device__ variable is the entire CUDA context in which it was defined. This means that any kernel or device function that runs under the same context can access that variable, regardless of where it was launched from.
3. The lifetime of a __device__ variable is tied to the lifetime of the CUDA context, which typically lasts for the duration of the application (or until the context is explicitly destroyed). Therefore, the variable persists across kernel launches and remains allocated until the application ends or the context is destroyed.
4. Unlike __shared__ variables (per block) or local variables (per thread), __device__ variables are global to the device and act like static storage.
5. Accessing or modifying a __device__ variable from host code requires special APIs (cudaMemcpyFromSymbol, cudaMemcpyToSymbol, or CUDA kernel functions). In this example we demonstrate reading and writing it from device code only.
6. Note that if multiple contexts are created (e.g., multiple cudaSetDevice calls or separate contexts in a multi-GPU program), each context has its own instance of the __device__ variable.
*/

#include <cstdio>
#include <cuda_runtime.h>

// Global device variable
__device__ int globalDeviceVar = 0;

// Kernel that demonstrates usage of the __device__ variable
__global__ void useDeviceVar()
{
    // Each thread reads the current value
    int val = globalDeviceVar;

    // For illustration, only thread 0 writes back
    if (threadIdx.x == 0)
    {
        // Increment the device variable
        globalDeviceVar = val + 1;
        printf("Device variable updated by thread %d: %d -> %d\n",
               threadIdx.x, val, globalDeviceVar);
    }
    else
    {
        // Other threads just read the value
        printf("Thread %d sees device variable: %d\n", threadIdx.x, globalDeviceVar);
    }
}

int main()
{
    // Initialize the device variable from host (optional)
    int initVal = 42;
    cudaMemcpyToSymbol(globalDeviceVar, &initVal, sizeof(int));

    // Launch kernel
    useDeviceVar<<<1, 4>>>();
    cudaDeviceSynchronize();

    // Check final value from host
    int finalVal;
    cudaMemcpyFromSymbol(&finalVal, globalDeviceVar, sizeof(int));
    printf("Final value of globalDeviceVar on host: %d\n", finalVal);

    return 0;
}
