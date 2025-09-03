/*
Aim: Explain what an atomic operation is (an indivisible operation that completes without interruption).

Thought process:
- The user wants an explanation of atomic operations, but the developer instruction demands we produce a CUDA C program (.cu file).
- The aim of the program is explicitly stated in the prompt; we must include that text verbatim in a multiline comment at the top of the file.
- To satisfy the developer instruction, we add our entire thinking inside the comment as well.
- The program itself will demonstrate an atomic operation: summing an array of integers on the GPU using atomicAdd to avoid race conditions.
- The code will be a minimal, self-contained CUDA C program that can be compiled with nvcc.
- We will allocate an array on the host, copy it to the device, run a kernel that atomically adds each element to a global sum, copy the result back, and print it.
- The code also contains some comments to explain each part.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define ARRAY_SIZE 1024

// Kernel that uses atomicAdd to safely accumulate array elements into a single global sum.
__global__ void atomicSumKernel(const int *input, int *output, int n)
{
    // Compute global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        // Atomically add the input element to the global output variable
        atomicAdd(output, input[idx]);
    }
}

int main(void)
{
    int *h_input = NULL;
    int *d_input = NULL;
    int *d_output = NULL;
    int h_output = 0;

    // Allocate host memory
    h_input = (int *)malloc(ARRAY_SIZE * sizeof(int));
    if (h_input == NULL)
    {
        fprintf(stderr, "Failed to allocate host input array.\n");
        return EXIT_FAILURE;
    }

    // Initialize host array with some values (e.g., 0 to ARRAY_SIZE-1)
    for (int i = 0; i < ARRAY_SIZE; ++i)
    {
        h_input[i] = i;
    }

    // Allocate device memory for input array
    cudaError_t err = cudaMalloc((void **)&d_input, ARRAY_SIZE * sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc d_input failed: %s\n", cudaGetErrorString(err));
        free(h_input);
        return EXIT_FAILURE;
    }

    // Allocate device memory for output (global sum)
    err = cudaMalloc((void **)&d_output, sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc d_output failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        free(h_input);
        return EXIT_FAILURE;
    }

    // Initialize device output to zero
    err = cudaMemset(d_output, 0, sizeof(int));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemset d_output failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        free(h_input);
        return EXIT_FAILURE;
    }

    // Copy input data from host to device
    err = cudaMemcpy(d_input, h_input, ARRAY_SIZE * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy HtoD failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        free(h_input);
        return EXIT_FAILURE;
    }

    // Launch kernel: one thread per array element
    int threadsPerBlock = 256;
    int blocksPerGrid = (ARRAY_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    atomicSumKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, ARRAY_SIZE);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        free(h_input);
        return EXIT_FAILURE;
    }

    // Wait for GPU to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        free(h_input);
        return EXIT_FAILURE;
    }

    // Copy the result back to host
    err = cudaMemcpy(&h_output, d_output, sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy DtoH failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_output);
        free(h_input);
        return EXIT_FAILURE;
    }

    // Compute expected sum on host for verification
    int expected = 0;
    for (int i = 0; i < ARRAY_SIZE; ++i)
    {
        expected += h_input[i];
    }

    // Print results
    printf("Atomic sum result from GPU: %d\n", h_output);
    printf("Expected sum (host calculation): %d\n", expected);
    if (h_output == expected)
    {
        printf("Success: Atomic operation produced correct result.\n");
    }
    else
    {
        printf("Error: Result mismatch.\n");
    }

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);

    return EXIT_SUCCESS;
}
