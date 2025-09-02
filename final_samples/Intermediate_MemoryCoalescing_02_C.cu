```cuda
/*
Aim of the program (word for word):
Phase 3: Coalesced write to `output`. Thread `(tx,ty)` reads from `tile[tx][ty]` (the transposed location) and writes to the transposed global position in `output`.

Thinking:
1. We implement a classic matrix transpose using shared memory tiling for coalesced memory accesses.
2. The kernel reads a BLOCK_DIM x BLOCK_DIM tile from global memory into a shared array `tile`.  
   We pad the second dimension by +1 to avoid shared‑memory bank conflicts.
3. After synchronizing, each thread writes the element from `tile[ty][tx]` to the transposed
   position in the output array.  
   The output index is computed with swapped block indices:  
   `out_row = blockIdx.x * BLOCK_DIM + ty` and `out_col = blockIdx.y * BLOCK_DIM + tx`.  
   This results in a coalesced write because threads in a warp write contiguous addresses.
4. The host code allocates input and output arrays, initializes the input with some data,
   copies it to the device, launches the kernel, copies the result back, and finally
   frees all memory.
5. Error checking is performed for CUDA API calls and kernel launch.
6. The program uses `float` for simplicity, but the kernel can be templated if needed.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define BLOCK_DIM 32   // Tile dimension (must be a multiple of 32 for best performance)

__global__ void transposeCoalesced(const float *input, float *output,
                                   int width, int height)
{
    // Shared memory tile with padding to avoid bank conflicts
    __shared__ float tile[BLOCK_DIM][BLOCK_DIM + 1];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global indices for reading from input
    int row = blockIdx.y * BLOCK_DIM + ty;
    int col = blockIdx.x * BLOCK_DIM + tx;

    // Read into shared memory
    if (row < height && col < width)
        tile[ty][tx] = input[row * width + col];

    __syncthreads();

    // Global indices for writing to output (transposed)
    int out_row = blockIdx.x * BLOCK_DIM + ty; // swapped block indices
    int out_col = blockIdx.y * BLOCK_DIM + tx;

    // Write back to output
    if (out_row < width && out_col < height)
        output[out_row * height + out_col] = tile[ty][tx];
}

// Utility macro for CUDA error checking
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n",         \
                    __FILE__, __LINE__, err, cudaGetErrorName(err), cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

int main()
{
    const int width  = 1024;  // Number of columns
    const int height = 1024;  // Number of rows

    size_t size_in_bytes = width * height * sizeof(float);

    // Allocate host memory
    float *h_input  = (float*)malloc(size_in_bytes);
    float *h_output = (float*)malloc(size_in_bytes); // output will have same size but transposed

    // Initialize input with some data (e.g., linear indices)
    for (int i = 0; i < width * height; ++i)
        h_input[i] = (float)i;

    // Allocate device memory
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc((void**)&d_input,  size_in_bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_output, size_in_bytes));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size_in_bytes, cudaMemcpyHostToDevice));

    // Define grid and block dimensions
    dim3 dimBlock(BLOCK_DIM, BLOCK_DIM);
    dim3 dimGrid((width  + BLOCK_DIM - 1) / BLOCK_DIM,
                 (height + BLOCK_DIM - 1) / BLOCK_DIM);

    // Launch kernel
    transposeCoalesced<<<dimGrid, dimBlock>>>(d_input, d_output, width, height);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size_in_bytes, cudaMemcpyDeviceToHost));

    // Simple verification: check a few elements
    bool ok = true;
    for (int i = 0; i < 5 && ok; ++i) {
        for (int j = 0; j < 5 && ok; ++j) {
            int idx_in  = i * width + j;
            int idx_out = j * height + i; // transposed
            if (h_input[idx_in] != h_output[idx_out]) {
                printf("Mismatch at (%d,%d): %f != %f\n", i, j, h_input[idx_in], h_output[idx_out]);
                ok = false;
            }
        }
    }
    if (ok)
        printf("Transpose successful!\n");

    // Clean up
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    free(h_input);
    free(h_output);

    return 0;
}
```