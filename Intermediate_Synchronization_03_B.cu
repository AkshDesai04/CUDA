/*
Aim of the program: Launch the kernel with 256 threads. The expected result is 256. What result do you actually get? (Something much smaller).

Thinking:
The task is to demonstrate that when a kernel is launched with 256 threads and each thread attempts to increment a single global integer without proper synchronization, the final value of that integer will be less than 256 due to race conditions. 
We will allocate a single integer on the device, initialize it to zero, and then launch a kernel with 256 threads. Each thread will perform a non-atomic increment of this integer: `*result += 1;`. 
Because this operation is not atomic, multiple threads may read, increment, and write the value simultaneously, overwriting each other's increments. The result after kernel execution will therefore be smaller than the expected 256.
After copying the result back to the host, we will print both the expected value and the actual value observed. 
This simple program illustrates the importance of using atomic operations when multiple threads write to the same memory location.

The program is written in CUDA C, compiled with `nvcc`. It includes basic error checking for CUDA API calls.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// Kernel that increments a single global integer without atomic operations
__global__ void incrementKernel(int *result) {
    // Each thread increments the result
    *result += 1;
}

int main(void) {
    int *d_result;
    int h_result = 0;
    const int expected = 256;

    // Allocate memory on device for result
    CUDA_CHECK(cudaMalloc((void**)&d_result, sizeof(int)));

    // Initialize device result to zero
    CUDA_CHECK(cudaMemset(d_result, 0, sizeof(int)));

    // Launch kernel with 256 threads in one block
    dim3 blockSize(256);
    dim3 gridSize(1);
    incrementKernel<<<gridSize, blockSize>>>(d_result);

    // Wait for GPU to finish before accessing on host
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(&h_result, d_result, sizeof(int), cudaMemcpyDeviceToHost));

    // Print expected vs actual
    printf("Expected result: %d\n", expected);
    printf("Actual result:   %d\n", h_result);

    // Clean up
    CUDA_CHECK(cudaFree(d_result));

    return 0;
}
