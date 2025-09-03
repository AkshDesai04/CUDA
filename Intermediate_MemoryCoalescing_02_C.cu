```cuda
/*
Phase 3: Coalesced write to `output`. Thread `(tx,ty)` reads from `tile[tx][ty]` (the transposed location) and writes to the transposed global position in `output`.

Thinking:
- The goal is to implement a matrix transpose kernel that performs coalesced global memory accesses.
- Each thread block will process a TILE_DIM x TILE_DIM tile of the matrix.
- Shared memory is used to avoid non-coalesced reads and to perform the transposition efficiently.
- A common technique is to add a padding byte/element to avoid shared memory bank conflicts. Here we use a +1 padding on the second dimension.
- Phase 3 specifically describes that after loading the tile into shared memory, each thread reads from `tile[tx][ty]` (note that this index refers to the transposed location) and writes to the global output at the transposed coordinates.
- In the kernel:
  * Each thread loads an element from global input into shared memory: `tile[tx][ty] = idata[input_idx]`.
  * After a __syncthreads() barrier, each thread writes that element to output at the transposed location:
    `odata[output_idx] = tile[tx][ty]`.
  * The output index is computed so that consecutive threads write to consecutive memory locations (coalesced).
- The host code sets up a simple test: allocate a small matrix, initialize it, copy to device, launch the kernel, copy back, and print a few values to verify correctness.
- Error checking macros are added for CUDA API calls.
*/

#include <stdio.h>
#include <stdlib.h>

#define TILE_DIM 32
#define TILE_DIM_PAD (TILE_DIM + 1)   // Padding to avoid bank conflicts

// Macro for checking CUDA errors
#define CUDA_CHECK(call)                                 \
    do {                                                 \
        cudaError_t err = call;                          \
        if (err != cudaSuccess) {                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__,                  \
                    cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                          \
        }                                                \
    } while (0)

// Kernel performing coalesced write to output after transposing tile indices
__global__ void transposeCoalesce(float *odata, const float *idata, int width, int height)
{
    __shared__ float tile[TILE_DIM][TILE_DIM_PAD];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;   // Global x coordinate
    int y = blockIdx.y * TILE_DIM + threadIdx.y;   // Global y coordinate

    // Boundary check
    if (x < width && y < height) {
        // Load element from global memory into shared memory
        tile[threadIdx.x][threadIdx.y] = idata[y * width + x];
    }

    __syncthreads();

    // Compute transposed global coordinates
    int transposed_x = blockIdx.y * TILE_DIM + threadIdx.x;
    int transposed_y = blockIdx.x * TILE_DIM + threadIdx.y;

    // Write back to global memory (coalesced)
    if (transposed_x < height && transposed_y < width) {
        odata[transposed_y * height + transposed_x] = tile[threadIdx.x][threadIdx.y];
    }
}

// Host function to initialize matrix
void initMatrix(float *mat, int width, int height)
{
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            mat[i * width + j] = (float)(i * width + j);
        }
    }
}

// Host function to print a submatrix
void printMatrix(const float *mat, int width, int height, int max_rows, int max_cols)
{
    for (int i = 0; i < max_rows && i < height; ++i) {
        for (int j = 0; j < max_cols && j < width; ++j) {
            printf("%6.1f ", mat[i * width + j]);
        }
        printf("\n");
    }
    printf("\n");
}

int main()
{
    const int width = 256;   // Matrix width
    const int height = 256;  // Matrix height

    size_t size = width * height * sizeof(float);

    // Allocate host memory
    float *h_input = (float *)malloc(size);
    float *h_output = (float *)malloc(size);

    // Initialize input matrix
    initMatrix(h_input, width, height);

    // Allocate device memory
    float *d_input = nullptr;
    float *d_output = nullptr;
    CUDA_CHECK(cudaMalloc((void **)&d_input, size));
    CUDA_CHECK(cudaMalloc((void **)&d_output, size));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    // Define block and grid dimensions
    dim3 blockDim(TILE_DIM, TILE_DIM);
    dim3 gridDim((width + TILE_DIM - 1) / TILE_DIM,
                 (height + TILE_DIM - 1) / TILE_DIM);

    // Launch kernel
    transposeCoalesce<<<gridDim, blockDim>>>(d_output, d_input, width, height);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    // Print a small portion of the matrices to verify
    printf("Input matrix (subset):\n");
    printMatrix(h_input, width, height, 8, 8);

    printf("Transposed matrix (subset):\n");
    printMatrix(h_output, height, width, 8, 8);  // Note: dimensions swapped after transpose

    // Free resources
    free(h_input);
    free(h_output);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
```