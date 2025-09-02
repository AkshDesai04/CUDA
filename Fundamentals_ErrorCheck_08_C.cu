/*
Aim: Run the version without error checking under `cuda-memcheck`. Observe that `cuda-memcheck` finds the error even if the program itself doesn't report one.

Thinking: The goal is to create a CUDA program that deliberately triggers a memory error, such as an out-of-bounds write. We will allocate a small array on the device and launch a kernel with more threads than array elements. The kernel writes to device memory based on thread index. Since there's no error checking after CUDA calls, the program itself won't report the error. However, cuda-memcheck should detect the out-of-bounds memory access. This demonstrates the usefulness of cuda-memcheck.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void writeKernel(int *d_arr, int size)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // Intentionally no bounds check to trigger out-of-bounds write
    d_arr[idx] = idx;
}

int main()
{
    const int N = 100;            // Number of elements in array
    const int BLOCK_SIZE = 32;    // Threads per block
    const int NUM_BLOCKS = 4;     // Number of blocks => 128 threads > N

    int *h_arr = (int*)malloc(N * sizeof(int));
    int *d_arr;

    cudaMalloc((void**)&d_arr, N * sizeof(int));

    // Launch kernel without error checking
    writeKernel<<<NUM_BLOCKS, BLOCK_SIZE>>>(d_arr, N);

    // Copy back results (ignoring potential errors)
    cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print first 10 elements
    for (int i = 0; i < 10; ++i)
    {
        printf("h_arr[%d] = %d\n", i, h_arr[i]);
    }

    // Clean up
    cudaFree(d_arr);
    free(h_arr);

    return 0;
}
