/*
Aim of the program (as passed in the prompt):
"Launch a 4x4 grid. Pass a 2D array (flattened to 1D) `int* data`. Have each thread write its flattened 1D block index to the corresponding location in the array."

Thinking behind the implementation:
- A 4x4 grid of thread blocks is required. The prompt says "each thread write its flattened 1D block index to the corresponding location in the array". This implies that each block has exactly one thread that writes the block's index.
- With one thread per block, the linear index into the flattened 2D array is identical to the block's linear index: `blockIdx.x + blockIdx.y * gridDim.x`.
- The kernel simply writes this value into the data array at that index.
- The host code allocates a 1D array of size 16 on the host and device, launches the kernel with the specified grid configuration, copies the data back, and prints the results to verify correctness.
- The code uses basic CUDA API calls: `cudaMalloc`, `cudaMemcpy`, and `cudaFree`.
- Error checking is minimal for brevity but can be extended in a production environment.

The program compiles with `nvcc` and runs on any CUDA-capable device.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void writeBlockIndex(int* data) {
    // Compute the flattened 1D index of this block
    int idx = blockIdx.x + blockIdx.y * gridDim.x;
    // Write the index value into the corresponding location in the array
    data[idx] = idx;
}

int main() {
    const int gridX = 4;
    const int gridY = 4;
    const int numBlocks = gridX * gridY;
    const int bytes = numBlocks * sizeof(int);

    // Host array
    int h_data[numBlocks];

    // Device array
    int* d_data;
    cudaError_t err = cudaMalloc((void**)&d_data, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Launch kernel with 4x4 grid of 1 thread per block
    dim3 grid(gridX, gridY);
    dim3 block(1);
    writeBlockIndex<<<grid, block>>>(d_data);

    // Wait for kernel to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 1;
    }

    // Copy result back to host
    err = cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_data);
        return 1;
    }

    // Free device memory
    cudaFree(d_data);

    // Print the result
    printf("Resulting array (flattened 1D):\n");
    for (int y = 0; y < gridY; ++y) {
        for (int x = 0; x < gridX; ++x) {
            int idx = x + y * gridX;
            printf("%2d ", h_data[idx]);
        }
        printf("\n");
    }

    return 0;
}
