```c
/*
Remember to use `cudaFreeHost` to free memory allocated with `cudaMallocHost`.

This program demonstrates the correct usage of `cudaMallocHost` and `cudaFreeHost`.  
The aim is to allocate pinned host memory, copy data to and from the GPU, and free the memory correctly using `cudaFreeHost`.  
Pinned memory allows faster DMA transfers between host and device.  
The program allocates an array of floats on the host using `cudaMallocHost`, initializes it, copies it to device memory, launches a simple kernel that multiplies each element by 2, copies the result back, prints a few values, and finally frees both the device memory (`cudaFree`) and the pinned host memory (`cudaFreeHost`).  
Error checking is performed after each CUDA API call and after the kernel launch to ensure robustness.  
The code follows C syntax with CUDA extensions and can be compiled with `nvcc` to produce an executable.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                     \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// Simple kernel that multiplies each element by 2
__global__ void multiplyByTwo(const float *in, float *out, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] * 2.0f;
    }
}

int main(void)
{
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(float);

    // Allocate pinned host memory
    float *h_data = NULL;
    CHECK_CUDA(cudaMallocHost((void **)&h_data, size));

    // Initialize host data
    for (int i = 0; i < N; ++i) {
        h_data[i] = (float)i;
    }

    // Allocate device memory
    float *d_in = NULL;
    float *d_out = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_in, size));
    CHECK_CUDA(cudaMalloc((void **)&d_out, size));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_in, h_data, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    multiplyByTwo<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, N);

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_data, d_out, size, cudaMemcpyDeviceToHost));

    // Verify a few results
    printf("Sample results after multiplication by 2:\n");
    for (int i = 0; i < 10; ++i) {
        printf("h_data[%d] = %f\n", i, h_data[i]);
    }

    // Clean up device memory
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));

    // Clean up pinned host memory
    CHECK_CUDA(cudaFreeHost(h_data));

    printf("Program completed successfully.\n");
    return 0;
}
```