```cpp
/*
Aim: Run `cuda-memcheck` with the `--tool racecheck` option to look for race conditions.

Thinking:
The goal is to provide a minimal CUDA program that deliberately introduces a race condition
so that when the user runs `cuda-memcheck --tool racecheck ./a.out` they will see
the race condition diagnostics.  A simple way to create a race is to have many threads
write to the same memory location without synchronization.  Here we launch a kernel
with a large number of threads that all increment the same element of an array.
The array is allocated on the device and also a host copy is allocated for verification.
After the kernel completes we copy the result back to the host and print it.  The
output will be unpredictable because of the race.  The user can compile the file
with `nvcc -arch=sm_61 test_race.cu -o test_race` and run
`cuda-memcheck --tool racecheck ./test_race` to see the race reported.

The code is intentionally simple, contains no synchronization primitives such as
atomics or locks, and does not use shared memory, so it should exhibit a race
condition on any GPU architecture.

*/

// CUDA program that demonstrates a data race
#include <stdio.h>
#include <cuda_runtime.h>

#define ARRAY_SIZE  1024
#define THREADS_PER_BLOCK 256
#define NUM_BLOCKS  64

// Kernel that creates a race condition by having all threads write to the same array element
__global__ void raceKernel(int *dev_arr, int element_index)
{
    // Each thread increments the same array element
    int idx = element_index;  // All threads use the same index
    for (int i = 0; i < 10; ++i) {
        dev_arr[idx] += 1;    // Non-atomic write => race
    }
}

int main(void)
{
    int *dev_arr = NULL;
    int *host_arr = NULL;
    size_t size = ARRAY_SIZE * sizeof(int);

    // Allocate device memory
    cudaError_t err = cudaMalloc((void**)&dev_arr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Device memory allocation failed: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Allocate host memory
    host_arr = (int*)malloc(size);
    if (!host_arr) {
        fprintf(stderr, "Host memory allocation failed\n");
        cudaFree(dev_arr);
        return -1;
    }

    // Initialize device array to zeros
    err = cudaMemset(dev_arr, 0, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Device memory set failed: %s\n", cudaGetErrorString(err));
        cudaFree(dev_arr);
        free(host_arr);
        return -1;
    }

    // Launch kernel: all threads target element 0
    int target_element = 0;
    raceKernel<<<NUM_BLOCKS, THREADS_PER_BLOCK>>>(dev_arr, target_element);

    // Wait for kernel to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(dev_arr);
        free(host_arr);
        return -1;
    }

    // Copy result back to host
    err = cudaMemcpy(host_arr, dev_arr, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Device to host copy failed: %s\n", cudaGetErrorString(err));
        cudaFree(dev_arr);
        free(host_arr);
        return -1;
    }

    // Print the value of the target element
    printf("Value of element %d after kernel execution: %d\n", target_element, host_arr[target_element]);

    // Cleanup
    cudaFree(dev_arr);
    free(host_arr);

    return 0;
}
```