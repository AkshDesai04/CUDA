```cuda
/*
Aim of the program: 
Explain what a 'race condition' is in the context of CUDA. 
Write a kernel where multiple threads attempt to increment the same global memory location without atomics 
(`output[0] = output[0] + 1`). Run it and show that the final result is incorrect.

Thinking:
In CUDA, threads execute concurrently on the GPU. 
When multiple threads read, modify, and write a shared memory location without proper synchronization,
they can interleave operations in unpredictable ways. This leads to a *race condition*,
where the final value depends on the order of execution, not on the number of operations performed.

To illustrate this, we create a global array `output` of size 1 initialized to 0. 
We launch many threads; each thread performs a simple read-modify-write:
`output[0] = output[0] + 1;`. Because no atomic operation is used,
threads may read the same original value, increment it, and write back the same result,
losing some increments. The final host value will be less than the number of threads,
demonstrating the race condition.

We will:
1. Allocate device memory for `output`.
2. Set `output` to 0.
3. Launch a kernel with e.g. 256 threads per block and 10 blocks (2560 threads).
4. Copy `output` back to host and print the value.
5. The expected value (without race condition) is 2560.
   The printed value will be lower, showing the race condition in effect.

Note: The exact incorrect result may vary between runs due to scheduling.
*/

#include <stdio.h>
#include <cuda.h>

__global__ void raceKernel(int *output)
{
    // Each thread attempts to increment output[0] without atomic
    output[0] = output[0] + 1;
}

int main()
{
    const int ARRAY_SIZE = 1;
    const int NUM_THREADS = 256;
    const int NUM_BLOCKS = 10; // Total 2560 increments

    int *dev_output;
    int host_output = 0;

    // Allocate device memory
    cudaMalloc((void**)&dev_output, ARRAY_SIZE * sizeof(int));

    // Initialize device memory to 0
    cudaMemcpy(dev_output, &host_output, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel
    raceKernel<<<NUM_BLOCKS, NUM_THREADS>>>(dev_output);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(&host_output, dev_output, ARRAY_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    // Print results
    printf("Expected result (without race): %d\n", NUM_THREADS * NUM_BLOCKS);
    printf("Actual result after race:       %d\n", host_output);

    // Clean up
    cudaFree(dev_output);
    return 0;
}
```