```cpp
/* 
Implement a matrix transpose: `output[x * height + y] = input[y * width + x]`. 
Note this can have memory coalescing issues.

Thought process:
1. The naive transpose kernel accesses global memory in a way that causes non-coalesced reads and writes, leading to poor performance. 
2. To improve coalescing, we use shared memory tiling. Threads within a block load a TILE_DIM x TILE_DIM submatrix from the input into shared memory. 
   - By loading with the original row-major indices (`idata[y*width + x]`) we preserve coalesced reads because consecutive threads in a warp read consecutive memory locations. 
   - We pad the shared memory array with an extra column (`TILE_DIM+1`) to avoid shared memory bank conflicts when transposing within the tile. 
3. After synchronizing, each thread writes a transposed element from shared memory to the output matrix. The write indices are swapped (`x` and `y` are interchanged) which also becomes coalesced because each thread in a warp writes to consecutive addresses in the output. 
4. Boundary conditions are handled by checking bounds before reading or writing. 
5. Host code allocates input and output matrices, copies data to device, launches the kernel, copies back, and verifies the result. 
6. Error checking macros are used to catch CUDA errors. 
7. We keep the code simple and self-contained, using float matrices for illustration. 
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8

/* CUDA error checking macro */
#define CUDA_CHECK(call)                                                \
    do {                                                                \
        cudaError_t err = call;                                         \
        if (err != cudaSuccess) {                                       \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",           \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                         \
        }                                                               \
    } while (0)

/* Kernel that transposes a matrix using shared memory tiling */
__global__ void transpose_kernel(float *odata, const float *idata, int width, int height)
{
    /* Allocate shared memory tile with padding to avoid bank conflicts */
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    /* Global indices for reading */
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    /* Load from global memory to shared memory */
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = idata[y * width + x];
    }

    __syncthreads();

    /* Global indices for writing (transposed) */
    x = blockIdx.y * TILE_DIM + threadIdx.x;  // swap blockIdx.x and blockIdx.y
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    /* Write from shared memory to global memory */
    if (x < height && y < width) {  // Note: output has swapped dimensions
        odata[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}

/* Host function to verify correctness */
bool verify_transpose(const float *input, const float *output, int width, int height)
{
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float in_val = input[y * width + x];
            float out_val = output[x * height + y];
            if (in_val != out_val) {
                fprintf(stderr, "Mismatch at (%d,%d): in=%f out=%f\n",
                        x, y, in_val, out_val);
                return false;
            }
        }
    }
    return true;
}

int main()
{
    /* Matrix dimensions */
    const int width = 1024;   // number of columns in input
    const int height = 1024;  // number of rows in input

    size_t size_in = width * height * sizeof(float);
    size_t size_out = width * height * sizeof(float);  // output size is same (just transposed)

    /* Allocate host memory */
    float *h_in  = (float*)malloc(size_in);
    float *h_out = (float*)malloc(size_out);

    if (!h_in || !h_out) {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    /* Initialize input matrix with some values (e.g., row-major index) */
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            h_in[y * width + x] = static_cast<float>(y * width + x);
        }
    }

    /* Allocate device memory */
    float *d_in = nullptr;
    float *d_out = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_in,  size_in));
    CUDA_CHECK(cudaMalloc((void**)&d_out, size_out));

    /* Copy input to device */
    CUDA_CHECK(cudaMemcpy(d_in, h_in, size_in, cudaMemcpyHostToDevice));

    /* Configure kernel launch parameters */
    dim3 blockDim(TILE_DIM, BLOCK_ROWS);
    dim3 gridDim((width  + TILE_DIM - 1) / TILE_DIM,
                 (height + TILE_DIM - 1) / TILE_DIM);

    /* Launch transpose kernel */
    transpose_kernel<<<gridDim, blockDim>>>(d_out, d_in, width, height);
    CUDA_CHECK(cudaGetLastError());

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_out, d_out, size_out, cudaMemcpyDeviceToHost));

    /* Verify result */
    if (verify_transpose(h_in, h_out, width, height)) {
        printf("Transpose successful!\n");
    } else {
        printf("Transpose failed!\n");
    }

    /* Clean up */
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);
    free(h_out);

    return 0;
}
```