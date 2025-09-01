/*
Write a kernel that has a 1-in-a-million chance of accessing out of bounds. Run it in a loop. Without error checking, it might run correctly 99% of the time, highlighting the danger of silent errors.

Thinking:
The goal is to create a CUDA program that demonstrates how subtle memory errors can remain undetected if we don't perform proper error checking. We need a kernel that occasionally (with a probability of 1/1,000,000) writes past the bounds of a device array. The kernel will run many times in a loop so that the out-of-bounds access is likely to happen at least once during execution. However, because the access is so rare and we do not call any CUDA error checking functions, the program may often appear to work correctly, masking the silent bug.

Implementation details:
1. Allocate a device array of a modest size (e.g., 1,000,000 integers). This gives enough room for normal writes but also makes the out-of-bounds index easy to generate (just write to index N).
2. The kernel uses the global thread ID and a simple pseudo-random generator based on a linear congruential formula to decide whether to write out of bounds. The probability is controlled by checking whether the generated number modulo 1,000,000 equals zero.
3. The kernel writes the current iteration number to the selected index. When the out-of-bounds write occurs, it will write past the end of the array, which is undefined behavior. In practice, the write may corrupt memory, crash, or silently corrupt data.
4. The host code launches the kernel in a loop for many iterations (e.g., 10,000). The grid and block configuration is chosen to keep the number of threads per launch small (e.g., 256 threads per block, 4 blocks). This ensures that the probability of an out-of-bounds write per launch is about 0.001024, giving roughly 10 occurrences over the entire run.
5. After the loop, we copy the array back to host and print the first few elements. No error checking is performed after any CUDA call; we simply ignore the return codes. This demonstrates how silent errors can escape detection.
6. The code is written in plain C++ with CUDA, and is fully self-contained. It can be compiled with nvcc and executed on any compatible device.

Overall, the program is intentionally unsafe to illustrate the danger of silent errors. Users should always check CUDA error codes after kernel launches and memory operations. This example shows how a seemingly innocuous bug can persist unnoticed if error checks are omitted.
*/

#include <stdio.h>
#include <stdlib.h>

#define ARRAY_SIZE 1000000          // Size of the device array
#define BLOCK_SIZE 256              // Threads per block
#define GRID_SIZE 4                 // Number of blocks
#define NUM_ITERATIONS 10000        // How many times to launch the kernel
#define PROBABILITY_DIVISOR 1000000 // 1 in 1,000,000 chance

__global__ void unsafeWriteKernel(int *data, int n, int iteration) {
    // Global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Simple linear congruential generator for pseudo-randomness
    // This generator is deterministic but gives enough variation per thread
    unsigned int seed = (unsigned int)(idx + iteration * 31);
    seed = seed * 1103515245 + 12345;
    unsigned int rnd = seed % PROBABILITY_DIVISOR;

    // Target index: usually the thread's own idx, but occasionally out-of-bounds
    int target = idx;
    if (rnd == 0) {  // 1 in 1,000,000 chance
        target = n;  // deliberately out-of-bounds write
    }

    // Perform the write
    data[target] = iteration;
}

int main(void) {
    int *h_array = (int *)malloc(ARRAY_SIZE * sizeof(int));
    if (!h_array) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // Initialize host array to zero
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        h_array[i] = 0;
    }

    int *d_array = NULL;
    // Allocate device memory
    cudaMalloc((void **)&d_array, ARRAY_SIZE * sizeof(int));

    // Copy host array to device (initially zero)
    cudaMemcpy(d_array, h_array, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);

    // Run kernel in a loop without error checking
    for (int iter = 0; iter < NUM_ITERATIONS; ++iter) {
        unsafeWriteKernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_array, ARRAY_SIZE, iter);
        // Intentionally omit cudaGetLastError() and cudaDeviceSynchronize()
    }

    // Copy back to host (synchronization occurs implicitly)
    cudaMemcpy(h_array, d_array, ARRAY_SIZE * sizeof(int), cudaMemcpyDeviceToHost);

    // Print first few elements to observe potential corruption
    printf("First 10 elements after kernel executions:\n");
    for (int i = 0; i < 10; ++i) {
        printf("h_array[%d] = %d\n", i, h_array[i]);
    }

    // Clean up
    cudaFree(d_array);
    free(h_array);

    return 0;
}
