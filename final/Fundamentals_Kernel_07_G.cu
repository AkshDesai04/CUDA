/*
Launch an 8x8 block. Pass a 2D array (flattened to 1D) `int* data`. Have each thread write its flattened 1D index to the corresponding location in the array.

Thinking process:
- The program should create a CUDA kernel that runs on a single block of dimensions 8x8.
- Each thread calculates its unique 1D index within the block using the formula:
    idx = threadIdx.y * blockDim.x + threadIdx.x
- Since there is only one block, the global index is the same as the local index.
- The thread writes this index value into the corresponding position of the flattened array `data[idx]`.
- The host code allocates memory for the array on both host and device, launches the kernel with one block of 8x8 threads, copies the result back, and prints the array to verify correctness.
- Basic CUDA error checking is included to make debugging easier.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// Kernel that writes the flattened index of each thread into the array
__global__ void writeIndices(int *data)
{
    // Calculate the 1D index for this thread within the block
    int idx = threadIdx.y * blockDim.x + threadIdx.x;

    // Write the index to the corresponding location in the array
    data[idx] = idx;
}

int main(void)
{
    const int rows = 8;
    const int cols = 8;
    const int size = rows * cols;          // 64
    const int bytes = size * sizeof(int);

    // Allocate host memory
    int *h_data = (int *)malloc(bytes);
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Allocate device memory
    int *d_data;
    CUDA_CHECK(cudaMalloc((void **)&d_data, bytes));

    // Launch kernel: 1 block of 8x8 threads
    dim3 blockDim(cols, rows, 1);
    dim3 gridDim(1, 1, 1);
    writeIndices<<<gridDim, blockDim>>>(d_data);
    CUDA_CHECK(cudaGetLastError());   // Check for launch errors

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));

    // Print the resulting array
    printf("Resulting array (flattened 1D):\n");
    for (int i = 0; i < size; ++i) {
        printf("%2d ", h_data[i]);
        if ((i + 1) % cols == 0) printf("\n");
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_data));
    free(h_data);

    return EXIT_SUCCESS;
}
