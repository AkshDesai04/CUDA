```cuda
/*
Implement an in-place version: data[i] = data[i] * data[i].
I will write a simple CUDA program that squares each element of an array in place.  
The kernel will be launched with enough threads to cover the array size, and it will
check bounds before performing the multiplication.  
In the host code I will allocate a small array of floats, initialize it with
consecutive values, copy it to the device, invoke the kernel, copy the data
back, and then print the before/after to confirm correctness.  
All CUDA API calls will be checked for errors.  The program is fully
self‑contained and can be compiled with nvcc.  
*/

// Include necessary headers
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// CUDA error checking macro
#define CHECK_CUDA(call)                                                  \
    do {                                                                  \
        cudaError_t err = call;                                           \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",  \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),      \
                    cudaGetErrorString(err));                            \
            exit(EXIT_FAILURE);                                          \
        }                                                                 \
    } while (0)

// Kernel: square each element in place
__global__ void squareKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= data[idx];
    }
}

int main(int argc, char *argv[]) {
    // Default array size
    const int n = (argc > 1) ? atoi(argv[1]) : 1024;

    // Allocate host array
    float *h_data = (float *)malloc(n * sizeof(float));
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host array with 1.0f, 2.0f, ... n
    for (int i = 0; i < n; ++i) {
        h_data[i] = static_cast<float>(i + 1);
    }

    // Print first few elements before squaring
    printf("Before squaring (first 10 elements):\n");
    for (int i = 0; i < (n > 10 ? 10 : n); ++i) {
        printf("%f ", h_data[i]);
    }
    printf("\n");

    // Allocate device memory
    float *d_data;
    CHECK_CUDA(cudaMalloc((void **)&d_data, n * sizeof(float)));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice));

    // Determine launch configuration
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    squareKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, n);
    CHECK_CUDA(cudaGetLastError());

    // Wait for GPU to finish
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost));

    // Print first few elements after squaring
    printf("After squaring (first 10 elements):\n");
    for (int i = 0; i < (n > 10 ? 10 : n); ++i) {
        printf("%f ", h_data[i]);
    }
    printf("\n");

    // Cleanup
    free(h_data);
    CHECK_CUDA(cudaFree(d_data));

    return 0;
}
```