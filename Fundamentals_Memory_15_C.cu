/*
Aim: What is the scope and lifetime of a __device__ variable? Explain in a comment. (Scope of the entire CUDA context, lifetime of the application).

Thinking:
- A __device__ variable declared at file scope (outside of any function) resides in device global memory.
- Its scope is the entire CUDA context: any kernel running on the device can access it, as long as it is compiled into the same binary or the symbol is made visible to the kernels.
- The variable is created when the CUDA context is initialized (typically when the first kernel launch or memory allocation occurs) and is destroyed when the context is destroyed, which happens when the CUDA application terminates or when cudaDeviceReset() is called.
- Therefore, the lifetime of a __device__ variable spans the whole application run time. The variable retains its value across multiple kernel launches, across different streams, and across multiple threads within the device.
- It can be accessed from the host via cudaMemcpyToSymbol / cudaMemcpyFromSymbol.
- The lifetime is independent of any particular kernel; it is not per-launch or per-thread.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// __device__ variable with file scope.
// It lives for the entire lifetime of the CUDA context.
__device__ int deviceCounter = 0;

// Simple kernel that increments the deviceCounter atomically.
__global__ void incrementCounter(int increments)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < increments)
    {
        atomicAdd(&deviceCounter, 1);
    }
}

int main()
{
    const int numIncrements = 1000;
    const int threadsPerBlock = 256;
    const int blocks = (numIncrements + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel to increment the deviceCounter.
    incrementCounter<<<blocks, threadsPerBlock>>>(numIncrements);
    cudaDeviceSynchronize();

    // Launch another kernel to add more increments.
    incrementCounter<<<blocks, threadsPerBlock>>>(numIncrements);
    cudaDeviceSynchronize();

    // Copy the value back to host.
    int hostCounter = 0;
    cudaMemcpyFromSymbol(&hostCounter, deviceCounter, sizeof(int), 0, cudaMemcpyDeviceToHost);

    printf("Final value of deviceCounter (should be %d): %d\n", 2 * numIncrements, hostCounter);

    // Reset the device (destroy context, thus deviceCounter is cleaned up).
    cudaDeviceReset();

    return 0;
}
