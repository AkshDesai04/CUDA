/*
After copying, use `cudaFree` to release the device memory.

Thoughts:
We need to write a complete CUDA C program that demonstrates allocation of device memory, copying data from host to device, running a simple kernel, copying back, and then releasing the device memory using cudaFree. 
We'll allocate a simple array of integers. 
We'll create a kernel that multiplies each element by 2. 
We'll set up error checking. 
After the kernel launch, we'll copy the results back to host. 
Then we will call cudaFree on the device pointer. 
Also free host memory. 
The code should compile with nvcc. 
We'll include <cuda_runtime.h> and <stdio.h>. 
We'll also include some basic error handling. 
The code will be in a .cu file. 
*/

#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_CUDA(call)                                                   \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Simple kernel that doubles each element
__global__ void doubleElements(int *d_data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_data[idx] *= 2;
    }
}

int main(void) {
    const int N = 1024;
    const size_t size = N * sizeof(int);

    // Allocate host memory
    int *h_data = (int*)malloc(size);
    if (h_data == NULL) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host data
    for (int i = 0; i < N; ++i) {
        h_data[i] = i;
    }

    // Allocate device memory
    int *d_data = NULL;
    CHECK_CUDA(cudaMalloc((void**)&d_data, size));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    // Launch kernel: 256 threads per block
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    doubleElements<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);

    // Check for kernel launch errors
    CHECK_CUDA(cudaGetLastError());

    // Copy results back to host
    CHECK_CUDA(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));

    // Release device memory
    CHECK_CUDA(cudaFree(d_data));

    // Print a few results
    printf("First 10 elements after doubling:\n");
    for (int i = 0; i < 10; ++i) {
        printf("%d ", h_data[i]);
    }
    printf("\n");

    // Free host memory
    free(h_data);

    return EXIT_SUCCESS;
}
