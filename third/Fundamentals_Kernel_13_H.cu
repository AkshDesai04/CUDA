```cuda
/*
Aim of the program: 
To demonstrate how to use cudaMemset in the host code to initialize a device array 'a' with a specific value (e.g., 1) before launching a kernel. 
The program allocates two integer arrays on the device: 'a' and 'b'. It uses cudaMemset to set each byte of array 'a' to 1, 
then launches a simple kernel that copies each element from 'a' to 'b' after adding 1 to it. Finally, the result is copied back 
to the host and printed. This example shows the interplay between host-side memory initialization and device-side processing. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 10

// Kernel that reads from array a, adds 1, and writes to array b
__global__ void add_one_kernel(const int *a, int *b, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        b[idx] = a[idx] + 1;
    }
}

// Helper macro for CUDA error checking
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",     \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),          \
                    cudaGetErrorString(err));                               \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

int main(void)
{
    int *d_a = NULL;   // Device array a
    int *d_b = NULL;   // Device array b
    int h_b[N];        // Host array to receive result

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&d_a, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_b, N * sizeof(int)));

    // Initialize device array a to byte value 1 using cudaMemset
    // This sets each byte of the array to 0x01, so each int will have value 0x01010101
    CUDA_CHECK(cudaMemset(d_a, 1, N * sizeof(int)));

    // Launch kernel: one block with N threads
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    add_one_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_b, d_b, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print the result
    printf("Result array (each element is original a + 1):\n");
    for (int i = 0; i < N; ++i)
    {
        printf("%d ", h_b[i]);
    }
    printf("\n");

    // Clean up
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));

    return 0;
}
```