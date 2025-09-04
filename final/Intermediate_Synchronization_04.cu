```cuda
/*
Aim: Fix the race condition from the previous problem by using an atomic operation: `atomicAdd(&output[0], 1)`. Verify that the result is now correct.

Thought Process:
- The previous problem likely involved multiple threads updating a shared counter without synchronization, leading to lost updates.
- To eliminate the race condition, we will replace the plain `output[0]++` with `atomicAdd(&output[0], 1)`. This atomic operation guarantees that each increment is performed safely even when many threads execute concurrently.
- The kernel will perform a fixed number of increments per thread. After the kernel completes, the host will check that the final value of `output[0]` matches the expected total (threads × increments per thread).
- We will also add basic error checking for CUDA API calls and print the result to confirm correctness.

The program below:
1. Allocates a single-element array on the device (`output`).
2. Launches a kernel that performs 1,000,000 increments per thread across many blocks.
3. Uses `atomicAdd` to safely update `output[0]`.
4. Copies the result back to the host and verifies the count.
5. Prints the expected vs. actual result and a success message.

Compile with:
    nvcc -o atomic_example atomic_example.cu
Run with:
    ./atomic_example
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define NUM_BLOCKS  256
#define THREADS_PER_BLOCK  256
#define INCREMENTS_PER_THREAD  1000

// Kernel that uses atomicAdd to increment a global counter
__global__ void counterKernel(int *output) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Each thread increments the counter INCREMENTS_PER_THREAD times
    for (int i = 0; i < INCREMENTS_PER_THREAD; ++i) {
        atomicAdd(output, 1);
    }
}

// Helper macro for CUDA error checking
#define CHECK_CUDA(call)                                    \
    do {                                                    \
        cudaError_t err = call;                             \
        if (err != cudaSuccess) {                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",    \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                             \
        }                                                   \
    } while (0)

int main(void) {
    int *d_output;
    int h_output = 0;
    size_t size = sizeof(int);

    // Allocate device memory for the output counter
    CHECK_CUDA(cudaMalloc((void**)&d_output, size));
    // Initialize counter to 0
    CHECK_CUDA(cudaMemset(d_output, 0, size));

    // Launch kernel
    counterKernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(d_output);
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(&h_output, d_output, size, cudaMemcpyDeviceToHost));

    // Expected result
    long long expected = (long long)NUM_BLOCKS * THREADS_PER_BLOCK * INCREMENTS_PER_THREAD;

    printf("Expected counter value: %lld\n", expected);
    printf("Actual counter value:   %d\n", h_output);

    if (h_output == expected) {
        printf("Result is correct. Race condition resolved.\n");
    } else {
        printf("Result is incorrect. There is still a race condition.\n");
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_output));

    return 0;
}
```