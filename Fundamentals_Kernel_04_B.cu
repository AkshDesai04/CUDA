```cuda
/*
Aim of the program:
Modify the program to launch with 32 blocks instead of 8.

Thinking:
The original kernel launch configuration used 8 blocks with a
certain number of threads per block. To increase parallelism and
demonstrate the modification, I will change the grid dimension
to 32 blocks. The number of threads per block remains the same
(256) to keep the total number of threads consistent with a
typical CUDA workload. The rest of the program (memory
allocation, kernel definition, error checking) stays the same.
The kernel simply writes the global thread index into an output
array. The host code allocates the array, launches the kernel,
copies the result back, and prints a few sample values to verify
execution.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define THREADS_PER_BLOCK 256
#define NUM_BLOCKS 32           // Changed from 8 to 32

__global__ void fillKernel(int *d_arr, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        d_arr[idx] = idx;
}

int main(void)
{
    const int N = THREADS_PER_BLOCK * NUM_BLOCKS; // Total number of elements
    size_t size = N * sizeof(int);

    // Allocate host memory
    int *h_arr = (int*)malloc(size);
    if (h_arr == NULL) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return 1;
    }

    // Allocate device memory
    int *d_arr;
    cudaError_t err = cudaMalloc((void**)&d_arr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory (error code %s)!\n", cudaGetErrorString(err));
        free(h_arr);
        return 1;
    }

    // Launch kernel
    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim(NUM_BLOCKS);
    fillKernel<<<gridDim, blockDim>>>(d_arr, N);

    // Check for launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch kernel (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        free(h_arr);
        return 1;
    }

    // Wait for GPU to finish before accessing on host
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to synchronize device (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        free(h_arr);
        return 1;
    }

    // Copy result back to host
    err = cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy memory from device to host (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        free(h_arr);
        return 1;
    }

    // Verify results (print first 10 values)
    printf("First 10 elements of the array:\n");
    for (int i = 0; i < 10 && i < N; ++i) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    // Cleanup
    cudaFree(d_arr);
    free(h_arr);
    return 0;
}
```