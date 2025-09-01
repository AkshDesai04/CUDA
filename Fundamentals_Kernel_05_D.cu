/*
Modify the formula for a 2D grid of 1D blocks. Launch with a grid of (4, 2) blocks, each with 16 threads. The global ID formula is more complex. Start by calculating a unique block ID: int blockId = blockIdx.y * gridDim.x + blockIdx.x;. Then calculate the global thread ID.
I decided to implement a CUDA program that demonstrates the calculation of a global thread ID for a 2D grid of 1D blocks. The kernel computes a unique block ID using the provided formula, then computes the global thread ID by multiplying that block ID by the number of threads per block and adding the local thread index. The result is stored in a global array. The host code allocates memory for this array, launches the kernel with a (4,2) grid and 16 threads per block, copies the results back to the host, and prints them. This showcases how to compute and use global thread IDs in a 2D grid configuration.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void computeGlobalID(int *out)
{
    // Compute a unique block ID for a 2D grid of 1D blocks
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;

    // Compute the global thread ID
    int globalId = blockId * blockDim.x + threadIdx.x;

    // Store the global ID in the output array
    // The index is the same as the global ID
    out[globalId] = globalId;
}

int main()
{
    const int threadsPerBlock = 16;
    const int gridX = 4;
    const int gridY = 2;
    const int numBlocks = gridX * gridY;
    const int totalThreads = numBlocks * threadsPerBlock;

    // Allocate host memory
    int *h_out = (int *)malloc(totalThreads * sizeof(int));
    if (h_out == NULL) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Allocate device memory
    int *d_out;
    cudaError_t err = cudaMalloc((void **)&d_out, totalThreads * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        free(h_out);
        return EXIT_FAILURE;
    }

    // Launch kernel
    dim3 blockDim(threadsPerBlock);
    dim3 gridDim(gridX, gridY);
    computeGlobalID<<<gridDim, blockDim>>>(d_out);

    // Check for kernel launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        free(h_out);
        return EXIT_FAILURE;
    }

    // Copy results back to host
    err = cudaMemcpy(h_out, d_out, totalThreads * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_out);
        free(h_out);
        return EXIT_FAILURE;
    }

    // Print the global IDs
    printf("Global thread IDs (total %d):\n", totalThreads);
    for (int i = 0; i < totalThreads; ++i) {
        printf("%d ", h_out[i]);
    }
    printf("\n");

    // Clean up
    cudaFree(d_out);
    free(h_out);

    return EXIT_SUCCESS;
}
