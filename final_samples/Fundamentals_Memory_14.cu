/*
Aim of the program: Allocate a device array. Pass its pointer to a kernel. Inside the kernel, deliberately write to an out-of-bounds index (e.g., d_ptr[BIG_NUMBER]). Run the program using cuda-memcheck and analyze its report to find the memory error.

Thinking:
1. Allocate a small device array (e.g., 5 integers) using cudaMalloc.
2. Write a simple kernel that receives the device pointer and writes to an index beyond the allocated range (e.g., index 10).
3. Launch the kernel with a single thread/block since we only need to trigger the out-of-bounds write.
4. Copy the device array back to the host and print a value to confirm the program runs.
5. Include comprehensive CUDA error checking to catch any API errors.
6. The code is self-contained and ready to compile with nvcc and run under cuda-memcheck to observe the memory error.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Macro for CUDA error checking
#define CHECK_CUDA(call)                                       \
    do {                                                       \
        cudaError_t err = call;                                \
        if (err != cudaSuccess) {                              \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",  \
                    #call, __FILE__, __LINE__,                 \
                    cudaGetErrorString(err));                  \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    } while (0)

// Kernel that writes out-of-bounds
__global__ void outOfBoundsKernel(int *d_ptr) {
    // Intentionally write to an index beyond the allocated size (size=5)
    int idx = 10; // BIG_NUMBER: out-of-bounds
    d_ptr[idx] = 42; // This should trigger a cuda-memcheck error
}

int main(void) {
    const int N = 5; // Size of the device array
    const int hostSize = N * sizeof(int);
    int *h_ptr = NULL;
    int *d_ptr = NULL;

    // Allocate host memory
    h_ptr = (int *)malloc(hostSize);
    if (h_ptr == NULL) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }
    // Initialize host array
    for (int i = 0; i < N; ++i) h_ptr[i] = i;

    // Allocate device memory
    CHECK_CUDA(cudaMalloc((void **)&d_ptr, hostSize));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_ptr, h_ptr, hostSize, cudaMemcpyHostToDevice));

    // Launch kernel with 1 block of 1 thread
    outOfBoundsKernel<<<1, 1>>>(d_ptr);

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());

    // Copy data back from device to host
    CHECK_CUDA(cudaMemcpy(h_ptr, d_ptr, hostSize, cudaMemcpyDeviceToHost));

    // Print host array contents
    printf("Host array after kernel launch:\n");
    for (int i = 0; i < N; ++i) {
        printf("h_ptr[%d] = %d\n", i, h_ptr[i]);
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_ptr));
    free(h_ptr);

    printf("Program completed. Run with 'cuda-memcheck ./a.out' to see the out-of-bounds error.\n");
    return EXIT_SUCCESS;
}
