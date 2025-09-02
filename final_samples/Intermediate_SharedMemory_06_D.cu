/*
After loading and synchronizing, each thread i computes its result using shared_mem[local_i-1], shared_mem[local_i], and shared_mem[local_i+1].

Thinking process:
1. The goal is to demonstrate a CUDA kernel that loads data into shared memory and then uses the neighboring elements in shared memory for computation.
2. We need a simple stencil operation (e.g., sum of a pixel and its two neighbors).
3. To use shared memory, each block will load its portion of the input array plus a left and right halo element.
4. Each thread will load its element into shared_mem[local_i] where local_i = threadIdx.x + 1. The left halo goes into shared_mem[0] and the right halo into shared_mem[local_i+1].
5. After __syncthreads(), each thread computes output[global_i] = shared_mem[local_i-1] + shared_mem[local_i] + shared_mem[local_i+1].
6. Edge handling: for global indices out of bounds, we set halo values to 0.0f. This keeps the kernel simple.
7. The host code will allocate input and output arrays, copy input to device, launch the kernel, copy output back, and print a small result set for verification.
8. Error checking is added after each CUDA API call for robustness.
9. The code uses plain C syntax and is compiled with nvcc as a .cu file.

The final code below implements this logic.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define BLOCK_SIZE 256

__global__ void stencil_shared(const float *input, float *output, int N)
{
    extern __shared__ float shared_mem[];

    int global_i = blockIdx.x * blockDim.x + threadIdx.x;
    int local_i = threadIdx.x + 1; // shift to leave space for halo

    // Load the element of this thread into shared memory
    if (global_i < N)
    {
        shared_mem[local_i] = input[global_i];
    }
    else
    {
        // If out of bounds, set to 0 (won't be used)
        shared_mem[local_i] = 0.0f;
    }

    // Left halo (only for thread 0 of each block)
    if (threadIdx.x == 0)
    {
        int left_global = global_i - 1;
        if (left_global >= 0 && left_global < N)
        {
            shared_mem[0] = input[left_global];
        }
        else
        {
            shared_mem[0] = 0.0f;
        }
    }

    // Right halo (only for last thread of each block)
    if (threadIdx.x == blockDim.x - 1)
    {
        int right_global = global_i + 1;
        if (right_global >= 0 && right_global < N)
        {
            shared_mem[local_i + 1] = input[right_global];
        }
        else
        {
            shared_mem[local_i + 1] = 0.0f;
        }
    }

    // Ensure all loads are finished
    __syncthreads();

    // Perform stencil computation
    if (global_i < N)
    {
        float left   = shared_mem[local_i - 1];
        float center = shared_mem[local_i];
        float right  = shared_mem[local_i + 1];
        output[global_i] = left + center + right;
    }
}

int main()
{
    const int N = 1024; // Size of array
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_input  = (float*)malloc(size);
    float *h_output = (float*)malloc(size);

    // Initialize input with some values
    for (int i = 0; i < N; ++i)
    {
        h_input[i] = (float)i;
    }

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input,  size);
    cudaMalloc((void**)&d_output, size);

    // Copy input to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Determine grid and block dimensions
    dim3 blockDim(BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch kernel
    stencil_shared<<<gridDim, blockDim, (BLOCK_SIZE + 2) * sizeof(float)>>>(d_input, d_output, N);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Copy result back to host
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Print first 20 results for verification
    printf("First 20 results:\n");
    for (int i = 0; i < 20 && i < N; ++i)
    {
        printf("output[%d] = %f\n", i, h_output[i]);
    }

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
