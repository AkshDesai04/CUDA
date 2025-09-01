```cuda
/*
Choose a block size of 256 threads. Calculate the required grid size for a vector of 1,048,576 elements.

Thinking:
- The user wants a CUDA program that demonstrates computing the grid size for a given vector length
  with a fixed block size of 256.
- The vector length is 1,048,576 (which is 2^20). With 256 threads per block, the exact number of
  blocks needed is 1,048,576 / 256 = 4096. However, we should compute it programmatically so
  that the code can handle any vector size.
- We'll implement a simple kernel that writes the global index into each element of the array.
- The host will:
  1. Define the vector size and block size.
  2. Compute the grid size using the ceiling division formula: (size + blockSize - 1) / blockSize.
  3. Allocate device memory, launch the kernel, copy back, and verify a few values.
  4. Print the computed grid size and confirm it matches 4096.
- This fulfills the requirement to calculate and display the grid size, while also demonstrating
  basic CUDA memory allocation and kernel launch.
*/

#include <stdio.h>
#include <cuda_runtime.h>

__global__ void initKernel(int *d_arr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_arr[idx] = idx;
    }
}

int main() {
    const int vectorSize = 1048576;  // 1,048,576 elements
    const int blockSize = 256;       // threads per block

    // Compute grid size: ceil(vectorSize / blockSize)
    int gridSize = (vectorSize + blockSize - 1) / blockSize;

    printf("Vector size: %d\n", vectorSize);
    printf("Block size (threads per block): %d\n", blockSize);
    printf("Computed grid size (number of blocks): %d\n", gridSize);

    // Allocate host memory
    int *h_arr = (int *)malloc(vectorSize * sizeof(int));
    if (!h_arr) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Allocate device memory
    int *d_arr = nullptr;
    cudaError_t err = cudaMalloc((void **)&d_arr, vectorSize * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        free(h_arr);
        return EXIT_FAILURE;
    }

    // Launch kernel
    initKernel<<<gridSize, blockSize>>>(d_arr, vectorSize);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        free(h_arr);
        return EXIT_FAILURE;
    }

    // Copy back to host
    err = cudaMemcpy(h_arr, d_arr, vectorSize * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_arr);
        free(h_arr);
        return EXIT_FAILURE;
    }

    // Verify a few elements
    bool ok = true;
    for (int i = 0; i < 10; ++i) {
        if (h_arr[i] != i) {
            fprintf(stderr, "Data mismatch at index %d: expected %d, got %d\n", i, i, h_arr[i]);
            ok = false;
            break;
        }
    }
    if (ok) {
        printf("First 10 elements verified successfully.\n");
    }

    // Cleanup
    cudaFree(d_arr);
    free(h_arr);

    return EXIT_SUCCESS;
}
```