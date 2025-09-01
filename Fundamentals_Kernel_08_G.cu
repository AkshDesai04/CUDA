/*
Aim of the program: Launch a 4x4 grid. Pass a 2D array (flattened to 1D) int* data. Have each thread write its flattened 1D block index to the corresponding location in the array.

Thinking:
1. The goal is to create a CUDA kernel that runs on a 4x4 grid of blocks.
2. Each block should contain exactly one thread (blockDim = 1x1) to avoid race conditions, since we want each block to write its own index.
3. The kernel will compute a flattened block index using `blockIdx.x + blockIdx.y * gridDim.x`.
4. That index will be written into the `data` array at the same position.
5. On the host side, we allocate an array of 16 integers, copy it to the device, launch the kernel, copy the result back, and print the values.
6. Basic error checking will be added for CUDA API calls.
7. The program is written in plain C with CUDA extensions (.cu file) and compiles with nvcc.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel: each thread writes its flattened block index to the corresponding position in the array.
__global__ void writeBlockIdx(int *data) {
    int idx = blockIdx.x + blockIdx.y * gridDim.x; // Flattened block index
    data[idx] = idx;
}

int main(void) {
    const int gridSizeX = 4;
    const int gridSizeY = 4;
    const int numElements = gridSizeX * gridSizeY;

    // Host array
    int h_data[numElements];
    for (int i = 0; i < numElements; ++i) {
        h_data[i] = -1; // Initialize with a sentinel value
    }

    // Device array
    int *d_data = NULL;
    cudaError_t err = cudaMalloc((void**)&d_data, numElements * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Copy initial data to device (optional, since kernel will overwrite)
    err = cudaMemcpy(d_data, h_data, numElements * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return EXIT_FAILURE;
    }

    // Launch kernel with 4x4 grid and 1x1 block
    dim3 gridDim(gridSizeX, gridSizeY);
    dim3 blockDim(1, 1);
    writeBlockIdx<<<gridDim, blockDim>>>(d_data);

    // Check for launch errors
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return EXIT_FAILURE;
    }

    // Wait for device to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return EXIT_FAILURE;
    }

    // Copy result back to host
    err = cudaMemcpy(h_data, d_data, numElements * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return EXIT_FAILURE;
    }

    // Print the results
    printf("Resulting array (flattened 1D):\n");
    for (int y = 0; y < gridSizeY; ++y) {
        for (int x = 0; x < gridSizeX; ++x) {
            int idx = x + y * gridSizeX;
            printf("%2d ", h_data[idx]);
        }
        printf("\n");
    }

    // Clean up
    cudaFree(d_data);
    return EXIT_SUCCESS;
}
