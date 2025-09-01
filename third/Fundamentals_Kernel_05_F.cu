/*
Launch with more threads than you need (e.g., for an array of size 100, launch 128 threads). Add a check `if (id < 100)` before printing or writing to memory.

My thinking: We need to demonstrate launching more threads than needed. We'll create an array of size 100 on the device. We'll launch 128 threads, maybe in one block of 128 threads. The kernel will compute a global thread ID, and only if id < 100 will it perform an operation such as write arr[id] = id or print a message. This will show that the extra threads (128-100 = 28) will not perform any memory writes or prints, preventing out-of-bounds writes. We'll also include basic error checking, allocate memory, copy back, and maybe print host results. We'll keep it simple, using printf from device. 
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define ARRAY_SIZE 100
#define THREADS_PER_BLOCK 128

__global__ void fillArray(int *d_arr, int size)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size)
    {
        d_arr[id] = id;
        printf("Thread %d writing to d_arr[%d] = %d\n", id, id, id);
    }
}

int main(void)
{
    int h_arr[ARRAY_SIZE];
    int *d_arr = NULL;
    cudaError_t err;

    // Allocate device memory
    err = cudaMalloc((void**)&d_arr, ARRAY_SIZE * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error allocating device memory: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Launch kernel with more threads than array size
    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim(1); // Only one block needed for this demo
    fillArray<<<gridDim, blockDim>>>(d_arr, ARRAY_SIZE);

    // Wait for kernel to finish and capture any launch errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        return 1;
    }

    // Copy result back to host
    err = cudaMemcpy(h_arr, d_arr, ARRAY_SIZE * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error copying data back to host: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        return 1;
    }

    // Print a few results
    printf("\nHost array contents (first 10 elements):\n");
    for (int i = 0; i < 10 && i < ARRAY_SIZE; ++i)
    {
        printf("h_arr[%d] = %d\n", i, h_arr[i]);
    }

    // Clean up
    cudaFree(d_arr);
    return 0;
}
