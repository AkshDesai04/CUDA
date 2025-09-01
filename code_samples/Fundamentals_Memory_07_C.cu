/*
Aim: Allocate memory with `cudaMallocManaged`. Now try to access this pointer from the host. Does it work? Why?

Thinking:
- `cudaMallocManaged` allocates *unified memory* that is accessible from both the CPU (host) and GPU (device) without explicit `cudaMemcpy` calls. 
- Because the pointer returned by `cudaMallocManaged` refers to memory that the CUDA driver tracks and automatically migrates between the host and device as needed, the host can directly read from or write to it. 
- When the GPU accesses this memory, pages are migrated to the GPU; when the CPU accesses it again, they are migrated back. 
- In this program we allocate an array of integers with `cudaMallocManaged`, initialize it from the host, launch a kernel that increments each element, and then read the results back from the host. The successful read demonstrates that the pointer can be used on the host side. 
- The program also includes basic error checking and prints diagnostic messages to explain why the host access is valid.
*/

#include <cuda_runtime.h>
#include <stdio.h>

/* Kernel that increments each element of the array */
__global__ void incrementKernel(int *data, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += 1;
    }
}

/* Helper macro for CUDA error checking */
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",       \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

int main(void)
{
    const int N = 10;
    int *devPtr = NULL;

    /* Allocate unified memory accessible from both host and device */
    CUDA_CHECK(cudaMallocManaged(&devPtr, N * sizeof(int)));

    /* Initialize the array on the host */
    printf("Initializing array on the host...\n");
    for (int i = 0; i < N; ++i) {
        devPtr[i] = i;
    }

    /* Print the initial values */
    printf("Initial array values (host): ");
    for (int i = 0; i < N; ++i) {
        printf("%d ", devPtr[i]);
    }
    printf("\n");

    /* Launch kernel to increment each element */
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    printf("Launching kernel to increment each element...\n");
    incrementKernel<<<blocks, threadsPerBlock>>>(devPtr, N);
    CUDA_CHECK(cudaGetLastError());

    /* Wait for GPU to finish */
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Read back the updated values on the host */
    printf("Updated array values after kernel execution (host): ");
    for (int i = 0; i < N; ++i) {
        printf("%d ", devPtr[i]);  // This works because unified memory is accessible from host
    }
    printf("\n");

    /* Clean up */
    CUDA_CHECK(cudaFree(devPtr));
    printf("Program completed successfully.\n");
    return 0;
}
