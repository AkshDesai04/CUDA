/*
The aim of the program is to demonstrate loading a tile from `input` into shared memory `s_tile` such that `s_tile[threadIdx.y][threadIdx.x] = input[...]`. This read is coalesced.

My thinking:
- The kernel uses a block size equal to the tile dimensions so that each warp accesses contiguous memory locations, guaranteeing coalesced global reads.
- Shared memory is declared as a 2‑D array `__shared__ float s_tile[TILE_DIM][TILE_DIM];`.
- Global indices are computed from `blockIdx` and `threadIdx`. Each thread loads one element: `s_tile[threadIdx.y][threadIdx.x] = input[row * width + col];`.
- Bounds checking ensures we don't read past the end of the input array.
- After loading the tile, the kernel writes the values back to an `output` array to demonstrate that the data was correctly transferred.
- The host code allocates input and output arrays, initializes the input with sample data, copies it to the device, launches the kernel with a grid that covers the whole matrix, copies the result back, and prints a few values for verification.
- Error checks after CUDA calls are included for robustness.
- A simple TILE_DIM of 16 is chosen, but it can be adjusted as needed.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define TILE_DIM 16
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,  \
                    cudaGetErrorString(err));                                \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

/*
 * Kernel: Load a tile from global memory into shared memory and copy it back to output.
 */
__global__ void tileCopyKernel(const float *input, float *output, int width, int height)
{
    // Allocate shared memory tile
    __shared__ float s_tile[TILE_DIM][TILE_DIM];

    // Compute global indices
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    int row = blockIdx.y * TILE_DIM + threadIdx.y;

    // Bounds check for reading input
    if (row < height && col < width) {
        // Coalesced read: threads in a warp read consecutive elements in a row
        s_tile[threadIdx.y][threadIdx.x] = input[row * width + col];
    } else {
        // Handle out-of-bounds by loading zero (optional)
        s_tile[threadIdx.y][threadIdx.x] = 0.0f;
    }

    // Synchronize to make sure the tile is fully loaded before using
    __syncthreads();

    // Optional: Do some work on the tile here (e.g., compute something)
    // For demonstration, we simply copy it back to output.

    // Bounds check for writing output
    if (row < height && col < width) {
        output[row * width + col] = s_tile[threadIdx.y][threadIdx.x];
    }
}

/*
 * Utility function to print a matrix (for debugging)
 */
void printMatrix(const float *mat, int width, int height, int maxRows = 8, int maxCols = 8)
{
    int rmax = (height < maxRows) ? height : maxRows;
    int cmax = (width < maxCols) ? width : maxCols;
    for (int r = 0; r < rmax; ++r) {
        for (int c = 0; c < cmax; ++c) {
            printf("%6.2f ", mat[r * width + c]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(void)
{
    // Matrix dimensions
    const int width  = 1024;
    const int height = 1024;
    const size_t size = width * height * sizeof(float);

    // Allocate host memory
    float *h_input  = (float *)malloc(size);
    float *h_output = (float *)malloc(size);
    if (!h_input || !h_output) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input with sample values
    for (int i = 0; i < width * height; ++i) {
        h_input[i] = (float)i;
    }

    // Allocate device memory
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc((void **)&d_input, size));
    CHECK_CUDA(cudaMalloc((void **)&d_output, size));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    // Define block and grid dimensions
    dim3 blockDim(TILE_DIM, TILE_DIM, 1);
    dim3 gridDim((width  + TILE_DIM - 1) / TILE_DIM,
                 (height + TILE_DIM - 1) / TILE_DIM,
                 1);

    // Launch kernel
    tileCopyKernel<<<gridDim, blockDim>>>(d_input, d_output, width, height);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    // Print a small portion of the output to verify correctness
    printf("Sample of output matrix:\n");
    printMatrix(h_output, width, height, 8, 8);

    // Clean up
    free(h_input);
    free(h_output);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    return 0;
}
