/*
Aim: Allocate memory with `cudaMallocManaged`. Now try to access this pointer from the host. Does it work? Why?

Thinking:
Unified memory created with `cudaMallocManaged` provides a single memory allocation that is accessible from both the host (CPU) and the device (GPU). The CUDA runtime manages page migration between CPU and GPU memory automatically, so the programmer does not need to perform explicit `cudaMemcpy` calls. When a kernel runs on the GPU, the CUDA driver will bring the required pages to the GPU; when the CPU later accesses the memory, the pages are migrated back to the CPU if necessary. Therefore, accessing the pointer from the host works seamlessly: the host can read/write the data just like a normal pointer. However, performance may suffer due to migration overhead if the data is frequently accessed by both host and device in an interleaved manner. In this example, we allocate a simple integer array, initialize it on the host, launch a kernel that increments each element, and then read the values back on the host to demonstrate that the memory is shared and accessible from both sides.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                 \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

__global__ void increment_kernel(int *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += 1;
    }
}

int main(void) {
    const int N = 10;
    int *managed_data = NULL;

    // Allocate unified memory accessible by both host and device
    CHECK_CUDA(cudaMallocManaged(&managed_data, N * sizeof(int)));

    // Initialize data on host
    for (int i = 0; i < N; ++i) {
        managed_data[i] = i;
    }

    // Launch kernel to increment each element
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    increment_kernel<<<blocksPerGrid, threadsPerBlock>>>(managed_data, N);
    CHECK_CUDA(cudaGetLastError());      // Check for launch errors
    CHECK_CUDA(cudaDeviceSynchronize()); // Wait for kernel to finish

    // Access the data again from the host
    printf("Values after kernel execution:\n");
    for (int i = 0; i < N; ++i) {
        printf("managed_data[%d] = %d\n", i, managed_data[i]);
    }

    // Clean up
    CHECK_CUDA(cudaFree(managed_data));
    return 0;
}
