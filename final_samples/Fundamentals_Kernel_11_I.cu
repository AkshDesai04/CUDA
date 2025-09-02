```cuda
/*
Aim of the program:
Use `atomicAdd` to safely have multiple threads increment a counter member within the single shared struct instance.

Thinking process:
1. Define a simple struct `CounterStruct` with an integer member `counter`.
2. Allocate an instance of this struct in shared memory so that all threads in a block can access the same object.
3. In the kernel, each thread performs `atomicAdd` on `sharedStruct.counter` to increment it safely without race conditions.
4. The kernel is launched with a sufficient number of threads such that the final counter value equals the number of threads.
5. After kernel execution, copy the result back to the host and print it to verify correctness.
6. Use `cudaMemcpyToSymbol` if we want to use a symbol in constant memory, but for simplicity we copy directly from device to host.
7. Ensure error checking for CUDA calls to aid debugging.

This demonstrates how `atomicAdd` ensures correct accumulation when many threads concurrently modify a shared data structure.
*/

#include <stdio.h>
#include <cuda_runtime.h>

struct CounterStruct {
    int counter;
};

__global__ void incrementKernel(CounterStruct *d_struct, int numIterations)
{
    // Use shared memory to store a single instance of CounterStruct for the block
    __shared__ CounterStruct sharedStruct;

    // Initialize shared struct once per block (by thread 0)
    if (threadIdx.x == 0) {
        sharedStruct.counter = 0;
    }
    __syncthreads();

    // Each thread performs a number of increments
    for (int i = 0; i < numIterations; ++i) {
        atomicAdd(&(sharedStruct.counter), 1);
    }

    __syncthreads();

    // After all increments, copy the result from shared memory to global memory
    if (threadIdx.x == 0) {
        d_struct->counter = sharedStruct.counter;
    }
}

int main(void)
{
    const int threadsPerBlock = 256;
    const int blocks = 1; // Only one block is needed for this demo
    const int numIterations = 1000; // Each thread will increment 1000 times

    // Allocate device memory for the struct
    CounterStruct *d_struct = nullptr;
    cudaMalloc((void**)&d_struct, sizeof(CounterStruct));

    // Launch kernel
    incrementKernel<<<blocks, threadsPerBlock>>>(d_struct, numIterations);

    // Wait for GPU to finish before accessing on host
    cudaDeviceSynchronize();

    // Copy result back to host
    CounterStruct h_struct;
    cudaMemcpy(&h_struct, d_struct, sizeof(CounterStruct), cudaMemcpyDeviceToHost);

    printf("Final counter value: %d\n", h_struct.counter);
    printf("Expected value: %d\n", threadsPerBlock * numIterations);

    // Clean up
    cudaFree(d_struct);
    cudaDeviceReset();

    return 0;
}
```