```cuda
/* 
 Aim:  To implement a CUDA kernel that performs a matrix transpose using
        shared memory.  After a __syncthreads(), the kernel writes the
        transposed element from shared memory to the output matrix at a
        position where the x and y thread indices are swapped
        (output[...] = s_tile[threadIdx.x][threadIdx.y]).  The write
        must be coalesced, which is achieved by computing the output
        linear index with swapped block indices and by padding the
        shared memory tile to avoid bank conflicts.

 Thinking:
 1. Choose a tile size that matches a warp (e.g., 32) to maximize
    throughput and simplify indexing.  We define TILE_DIM = 32 and
    use a shared memory array of size TILE_DIM x (TILE_DIM+1) to
    provide padding and avoid shared memory bank conflicts.

 2. For coalesced global memory reads, each thread loads one element
    from the input matrix into shared memory:
        s_tile[threadIdx.y][threadIdx.x] = idata[row * width + col];
    where row = blockIdx.y * TILE_DIM + threadIdx.y and
    col = blockIdx.x * TILE_DIM + threadIdx.x.

 3. After __syncthreads(), each thread writes back the transposed
    element from shared memory to the output matrix.  The output
    index is computed with swapped block indices:
        xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
        yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
        odata[yIndex * height + xIndex] = s_tile[threadIdx.x][threadIdx.y];
    This pattern guarantees that consecutive threads in a warp write
    to consecutive memory locations in the output array, ensuring
    coalesced writes.

 4. The host code allocates a sample matrix, copies it to the device,
    launches the kernel, and copies the transposed result back.
    A simple check verifies that the transpose was performed
    correctly.

 5. The code uses standard CUDA error checking and is fully
    self‑contained in a single .cu file, ready to compile with
    nvcc.

*/

#include <stdio.h>
#include <cuda_runtime.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8

// CUDA error checking macro
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",    \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

// Kernel: transpose a matrix using shared memory
__global__ void matrixTranspose(float *odata, const float *idata,
                                int width, int height)
{
    // Shared memory tile with padding to avoid bank conflicts
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    // Calculate global indices for the element to load
    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
    int index_in = yIndex * width + xIndex;

    // Coalesced read from global memory into shared memory
    if (xIndex < width && yIndex < height)
        tile[threadIdx.y][threadIdx.x] = idata[index_in];
    else
        tile[threadIdx.y][threadIdx.x] = 0.0f; // Padding for partial tiles

    __syncthreads();

    // Compute indices for the transposed element
    xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
    int index_out = yIndex * height + xIndex;

    // Coalesced write to global memory from transposed shared memory
    if (xIndex < height && yIndex < width)
        odata[index_out] = tile[threadIdx.x][threadIdx.y];
}

// Host helper to fill a matrix with sample data
void initMatrix(float *mat, int width, int height)
{
    for (int i = 0; i < height; ++i)
        for (int j = 0; j < width; ++j)
            mat[i * width + j] = (float)(i * width + j);
}

// Simple verification of transpose
bool verifyTranspose(const float *original, const float *transposed,
                     int width, int height)
{
    for (int i = 0; i < height; ++i)
        for (int j = 0; j < width; ++j)
            if (original[i * width + j] != transposed[j * height + i])
                return false;
    return true;
}

int main(void)
{
    const int width  = 1024; // Number of columns in original matrix
    const int height = 1024; // Number of rows in original matrix

    size_t size = width * height * sizeof(float);

    // Allocate host matrices
    float *h_input  = (float *)malloc(size);
    float *h_output = (float *)malloc(size);

    // Initialize input matrix
    initMatrix(h_input, width, height);

    // Allocate device matrices
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc((void **)&d_input,  size));
    CUDA_CHECK(cudaMalloc((void **)&d_output, size));

    // Copy input matrix to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    // Define block and grid dimensions
    dim3 dimBlock(TILE_DIM, BLOCK_ROWS);
    dim3 dimGrid((width  + TILE_DIM - 1) / TILE_DIM,
                 (height + TILE_DIM - 1) / TILE_DIM);

    // Launch the transpose kernel
    matrixTranspose<<<dimGrid, dimBlock>>>(d_output, d_input, width, height);
    CUDA_CHECK(cudaGetLastError());

    // Copy transposed matrix back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    // Verify the result
    if (verifyTranspose(h_input, h_output, width, height))
        printf("Transpose verified successfully.\n");
    else
        printf("Transpose verification failed.\n");

    // Cleanup
    free(h_input);
    free(h_output);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    return 0;
}
```