/*
Aim of the program:
Set up a 1D problem. Define `int N = 1024; int threads_per_block = 256;`. Calculate the number of blocks needed: `int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;`. Create `dim3` variables from this and launch.

Thinking:
The user wants a CUDA C program that demonstrates how to set up a 1D kernel launch configuration. 
I will create a simple kernel that writes the global thread index into an array.  
We will allocate memory on both host and device, launch the kernel with the computed grid and block dimensions, copy the result back, and print a few values to confirm correct execution.  
All code will be self‑contained in a single .cu file, with the requested comment block at the top, and no extraneous text outside of the code.  
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024
#define THREADS_PER_BLOCK 256

__global__ void fillArray(int *d_arr, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        d_arr[idx] = idx;
}

int main()
{
    int *h_arr = (int*)malloc(N * sizeof(int));
    int *d_arr;
    cudaMalloc((void**)&d_arr, N * sizeof(int));

    int blocks_per_grid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    dim3 dimBlock(THREADS_PER_BLOCK);
    dim3 dimGrid(blocks_per_grid);

    fillArray<<<dimGrid, dimBlock>>>(d_arr, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_arr, d_arr, N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("First 10 elements of the array:\n");
    for (int i = 0; i < 10; ++i)
        printf("%d ", h_arr[i]);
    printf("\n");

    cudaFree(d_arr);
    free(h_arr);
    return 0;
}