```cuda
/*
The naive transpose reads from `input` uncoalesced and writes coalesced. 
This version reads coalesced and writes coalesced.

== Reasoning ==
In a row-major 2‑D matrix, elements of the same row are stored contiguously.
To achieve coalesced reads we let consecutive threads in a warp read consecutive
columns of the same row, i.e. each thread reads element (row, col) where the
`col` index varies with `threadIdx.x`.  We achieve this by mapping the
thread indices to matrix indices as:

    x = blockIdx.x * TILE_DIM + threadIdx.x   // column
    y = blockIdx.y * TILE_DIM + threadIdx.y   // row
    index_in = y * width + x

This guarantees that the global memory accesses performed by threads of the
same warp are contiguous and thus coalesced.

For coalesced writes after transposition we swap the role of the block
indices and the thread indices when writing to the output matrix.
After loading the data into a shared‑memory tile, we write the transposed
tile to global memory with:

    x_out = blockIdx.y * TILE_DIM + threadIdx.x
    y_out = blockIdx.x * TILE_DIM + threadIdx.y
    index_out = y_out * height + x_out

Again, for a fixed `blockIdx.x` and `threadIdx.y`, the `x_out` index
varies with `threadIdx.x`, which means consecutive threads write to
consecutive columns of the same output row – a coalesced write pattern.

To avoid shared‑memory bank conflicts we pad the shared array to
TILE_DIM+1 columns.  Boundary checks are performed for matrices whose
dimensions are not multiples of the tile size.  A simple host test
generates a matrix, copies it to the device, runs the kernel, copies
the result back, and verifies correctness.  Timing is provided using
CUDA events.

== Implementation Notes ==
* TILE_DIM = 32 (typical warp size) – yields a 32×32 tile.
* The shared array is declared as `__shared__ float tile[TILE_DIM][TILE_DIM+1]`.
* The kernel launches with `dim3 grid((width + TILE_DIM-1)/TILE_DIM, (height + TILE_DIM-1)/TILE_DIM);`
  and `dim3 block(TILE_DIM, TILE_DIM);`.
* Error checking macro `CUDA_CHECK` wraps all CUDA API calls.
* Host matrix is allocated as a 1‑D array for simplicity.
* Matrix values are initialized to `row * width + col` to make verification easy.
* After transposition, the element at `(row, col)` in the output should be
  equal to `col * width + row` of the input.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_DIM 32
#define BLOCK_ROWS 32

// Error checking macro
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

// Kernel: coalesced read, coalesced write
__global__ void transpose(float *odata, const float *idata, int width, int height)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 to avoid bank conflicts

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int index_in = y * width + x;

    if (x < width && y < height)
        tile[threadIdx.y][threadIdx.x] = idata[index_in];
    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x; // transpose indices for write
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    int index_out = y * height + x;

    if (x < height && y < width)
        odata[index_out] = tile[threadIdx.x][threadIdx.y];
}

int main(int argc, char **argv)
{
    // Matrix dimensions (width x height)
    int width  = 1024;
    int height = 1024;

    size_t size = width * height * sizeof(float);
    float *h_input  = (float*)malloc(size);
    float *h_output = (float*)malloc(size);

    // Initialize input matrix
    for (int row = 0; row < height; ++row) {
        for (int col = 0; col < width; ++col) {
            h_input[row * width + col] = (float)(row * width + col);
        }
    }

    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc((void**)&d_input,  size));
    CUDA_CHECK(cudaMalloc((void**)&d_output, size));

    // Copy input to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    // Configure kernel launch
    dim3 dimBlock(TILE_DIM, TILE_DIM);
    dim3 dimGrid((width  + TILE_DIM - 1) / TILE_DIM,
                 (height + TILE_DIM - 1) / TILE_DIM);

    // Timing events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    transpose<<<dimGrid, dimBlock>>>(d_output, d_input, width, height);
    CUDA_CHECK(cudaEventRecord(stop));

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Kernel execution time: %f ms\n", milliseconds);

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    // Verify correctness
    int errors = 0;
    for (int row = 0; row < height && errors < 10; ++row) {
        for (int col = 0; col < width; ++col) {
            float expected = (float)(col * width + row); // original (row, col) -> transposed (col, row)
            float got = h_output[row * height + col];
            if (fabs(expected - got) > 1e-5) {
                printf("Mismatch at (%d,%d): expected %f, got %f\n",
                       row, col, expected, got);
                ++errors;
            }
        }
    }
    if (errors == 0)
        printf("Transpose verification passed.\n");
    else
        printf("Transpose verification failed with %d errors.\n", errors);

    // Clean up
    free(h_input);
    free(h_output);
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
```