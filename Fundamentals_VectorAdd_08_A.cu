/*
 * Implement an in-place version: `data[i] += scalar_value`.
 *
 * Thinking:
 * - The goal is to create a CUDA program that demonstrates adding a scalar value to each element
 *   of an array in place, i.e. modifying the array directly on the device without creating a
 *   new array.  
 * - We'll use a simple 1D kernel where each thread processes one element.  
 * - The kernel will accept a pointer to the device array, the scalar value, and the number of
 *   elements.  
 * - In the host code, we'll allocate an array on the host, initialize it, copy it to the device,
 *   launch the kernel, copy the results back, and verify the outcome by printing a few elements.  
 * - Basic error checking for CUDA API calls will be added to make the program robust.  
 * - The program is self‑contained and compiles to a .cu file.
 */

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                     \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

__global__ void add_scalar_inplace(float *data, float scalar, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        data[idx] += scalar;
    }
}

int main(void)
{
    const int N = 1024;                 // Number of elements
    const float scalar = 3.14f;         // Scalar to add
    size_t size = N * sizeof(float);

    // Allocate host memory and initialize
    float *h_data = (float *)malloc(size);
    if (!h_data)
    {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }
    for (int i = 0; i < N; ++i)
        h_data[i] = (float)i;           // Example initialization

    // Allocate device memory
    float *d_data = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_data, size));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    // Launch kernel: one thread per element
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    add_scalar_inplace<<<blocksPerGrid, threadsPerBlock>>>(d_data, scalar, N);
    CHECK_CUDA(cudaGetLastError());

    // Wait for GPU to finish before accessing on host
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));

    // Verify result by printing first 10 elements
    printf("First 10 elements after adding %.2f:\n", scalar);
    for (int i = 0; i < 10 && i < N; ++i)
        printf("h_data[%d] = %f\n", i, h_data[i]);

    // Clean up
    free(h_data);
    CHECK_CUDA(cudaFree(d_data));

    return 0;
}
