/*
Modify the program to allocate and free the memory in a loop 100 times.
This program demonstrates repeated allocation and deallocation of device memory in a loop.
The aim is to stress-test the allocation/deallocation mechanism and ensure correctness
without memory leaks. In each iteration we allocate an array of 1M integers on the
GPU, optionally perform a simple kernel launch, and then free the memory. We also
check for errors after each CUDA call. The code prints a message when allocation
fails or when a kernel launch fails. After completing the 100 iterations, the
program reports success.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void dummyKernel(int *d_arr, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        d_arr[idx] = idx;
}

int main(void)
{
    const int iterations = 100;
    const size_t N = 1 << 20;            // 1 million integers
    const size_t bytes = N * sizeof(int);

    for (int i = 0; i < iterations; ++i)
    {
        int *d_arr = NULL;
        cudaError_t err = cudaMalloc((void **)&d_arr, bytes);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "cudaMalloc failed at iteration %d: %s\n",
                    i, cudaGetErrorString(err));
            return 1;
        }

        dim3 block(256);
        dim3 grid((N + block.x - 1) / block.x);
        dummyKernel<<<grid, block>>>(d_arr, N);

        err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Kernel launch failed at iteration %d: %s\n",
                    i, cudaGetErrorString(err));
            cudaFree(d_arr);
            return 1;
        }

        err = cudaDeviceSynchronize();
        if (err != cudaSuccess)
        {
            fprintf(stderr, "cudaDeviceSynchronize failed at iteration %d: %s\n",
                    i, cudaGetErrorString(err));
            cudaFree(d_arr);
            return 1;
        }

        err = cudaFree(d_arr);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "cudaFree failed at iteration %d: %s\n",
                    i, cudaGetErrorString(err));
            return 1;
        }
    }

    printf("Completed %d iterations of allocation/deallocation successfully.\n", iterations);
    return 0;
}
