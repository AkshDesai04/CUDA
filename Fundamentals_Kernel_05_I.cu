```cuda
/*
Aim of the program: Calculate the global ID starting from 1 instead of 0.

Thinking process:
1. The task is to write a CUDA C program that demonstrates how to compute a thread's global ID
   but with the offset that IDs start at 1 rather than the default 0.  The global ID in a
   1‑D grid is normally
       global_id = blockIdx.x * blockDim.x + threadIdx.x
   which yields values 0,1,2,...  We simply add 1 to this value.
2. For clarity and simplicity we will launch a small number of threads (e.g., 10) in a
   single block.  The kernel will write each thread’s adjusted global ID into a device
   array.  After copying the array back to the host we print the values to verify that
   they start from 1.
3. We use standard CUDA runtime API functions: cudaMalloc, cudaMemcpy, cudaFree,
   cudaGetLastError, cudaDeviceSynchronize.  Basic error checking is performed.
4. The kernel is trivial – it calculates the adjusted global ID and stores it.
   Because the calculation is simple we keep the code minimal and readable.
5. The program is fully self‑contained and can be compiled with `nvcc` to produce a
   single executable.

Now the actual CUDA C code follows. 
*/
#include <stdio.h>
#include <cuda_runtime.h>

// Kernel that writes each thread's global ID (starting from 1) into an array.
__global__ void computeGlobalID(int *output, int stride)
{
    // Compute the standard 0‑based global ID.
    int gid0 = blockIdx.x * blockDim.x + threadIdx.x;
    // Adjust to start counting from 1.
    int gid1 = gid0 + 1;
    // Store in the output array. Use stride in case we want to scatter.
    output[gid0] = gid1;
}

int main()
{
    const int numThreads = 10;      // total number of threads
    const int threadsPerBlock = 10;
    const int blocksPerGrid = (numThreads + threadsPerBlock - 1) / threadsPerBlock;

    int *d_output;
    int *h_output = (int*)malloc(numThreads * sizeof(int));
    if (!h_output) {
        fprintf(stderr, "Host memory allocation failed\n");
        return 1;
    }

    // Allocate device memory
    cudaError_t err = cudaMalloc((void**)&d_output, numThreads * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "Device memory allocation failed: %s\n", cudaGetErrorString(err));
        free(h_output);
        return 1;
    }

    // Launch kernel
    computeGlobalID<<<blocksPerGrid, threadsPerBlock>>>(d_output, numThreads);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_output);
        free(h_output);
        return 1;
    }

    // Wait for GPU to finish before accessing on host
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_output);
        free(h_output);
        return 1;
    }

    // Copy result back to host
    err = cudaMemcpy(h_output, d_output, numThreads * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_output);
        free(h_output);
        return 1;
    }

    // Print the results
    printf("Global IDs starting from 1:\n");
    for (int i = 0; i < numThreads; ++i) {
        printf("Thread %d: ID = %d\n", i, h_output[i]);
    }

    // Clean up
    cudaFree(d_output);
    free(h_output);

    return 0;
}
```