```cuda
/*
What happens if you allocate memory but your program crashes before `cudaFree` is called? Does the memory get reclaimed?

My thinking: In CUDA, device memory is managed by the driver and is associated with the process. If a program allocates memory with `cudaMalloc` and then crashes (e.g., due to a segmentation fault or an explicit `exit` before calling `cudaFree`), the CUDA driver will eventually reclaim the memory when the process terminates. This is similar to how operating system memory is reclaimed when a process exits. However, if the process is terminated abruptly (e.g., by a kill signal or a crash that prevents the driver from being notified), the driver typically cleans up all resources associated with the process, including device memory. In practice, this means that not calling `cudaFree` before the program exits usually does not leave memory permanently allocated on the GPU; the driver will release it when the process ends. To demonstrate this behavior, the following CUDA program allocates some device memory, then intentionally exits before freeing it. When you run this program, you will observe that the device memory is reclaimed automatically upon program termination.*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Simple kernel that does nothing
__global__ void do_nothing() {
    // No operation
}

// Macro for error checking
#define CHECK_CUDA(call)                                      \
    do {                                                      \
        cudaError_t err = call;                               \
        if (err != cudaSuccess) {                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                               \
        }                                                     \
    } while (0)

int main(void) {
    size_t N = 1024 * 1024 * 10; // Allocate ~80MB (10 million ints)
    int *dev_ptr = NULL;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void**)&dev_ptr, N * sizeof(int)));

    // Launch kernel (does nothing)
    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    do_nothing<<<grid, block>>>();
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Intentionally exit before freeing device memory
    printf("Allocated %zu integers on device. Exiting without cudaFree.\n", N);
    // Normally, we would call cudaFree(dev_ptr); here, but we skip it intentionally.
    // Also skip cudaDeviceReset() to simulate abrupt termination.

    // Exit program
    exit(EXIT_SUCCESS);
}
```