```cuda
/* 
Aim of the program: 
    The goal is to transpose a 2‑D matrix stored in row‑major order using CUDA. 
    Each thread (x, y) reads element input[y*width + x] and writes it to 
    output[x*height + y]. The kernel is written to take advantage of 
    coalesced memory accesses for the read from input (threads in a warp 
    share the same y and consecutive x values). It also uses shared memory 
    tiling to avoid bank conflicts and to improve write coalescing for 
    the output. The program demonstrates host‑side memory allocation, 
    kernel launch, timing, and result verification.

Thinking about the solution:
    1. Choose block dimensions that are multiples of the warp size (32). 
       A typical choice is 32×32 or 16×16 tiles. 
    2. For coalesced reads: the kernel will be launched with a grid such 
       that threads in a block have consecutive x indices for a given y. 
       The global read stride will then be 1. 
    3. For coalesced writes: the transposed matrix stores columns in 
       contiguous memory. By loading a tile into shared memory and then 
       writing it back transposed, we ensure that threads write to 
       consecutive addresses. 
    4. Shared memory tiling also reduces global memory traffic and 
       mitigates bank conflicts by using a padding index. 
    5. We include simple error checking macros, a timing helper using 
       CUDA events, and a verification routine that compares the CUDA 
       result with a CPU reference implementation. 
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/* Helper macro for error checking */
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err = call;                                                   \
        if (err != cudaSuccess) {                                                 \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,      \
                    cudaGetErrorString(err));                                    \
            exit(EXIT_FAILURE);                                                   \
        }                                                                         \
    } while (0)

/* Kernel for transposing a matrix using shared memory tiling */
#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transpose(const float *in, float *out, int width, int height)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 to avoid bank conflicts

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    /* Read from global memory into shared memory (coalesced if same y) */
    if (x < width && y < height)
        tile[threadIdx.y][threadIdx.x] = in[y * width + x];

    __syncthreads();

    /* Write transposed data from shared memory to global memory (coalesced) */
    int transposed_x = blockIdx.y * TILE_DIM + threadIdx.x;
    int transposed_y = blockIdx.x * TILE_DIM + threadIdx.y;

    if (transposed_x < height && transposed_y < width)
        out[transposed_y * height + transposed_x] = tile[threadIdx.x][threadIdx.y];
}

/* CPU reference implementation for verification */
void cpu_transpose(const float *in, float *out, int width, int height)
{
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            out[x * height + y] = in[y * width + x];
}

/* Main driver */
int main(void)
{
    /* Matrix dimensions */
    int width = 1024;   // number of columns
    int height = 1024;  // number of rows

    size_t size_in = width * height * sizeof(float);
    size_t size_out = width * height * sizeof(float);

    /* Allocate host memory */
    float *h_in  = (float *)malloc(size_in);
    float *h_out = (float *)malloc(size_out);
    float *h_ref = (float *)malloc(size_out);

    /* Initialize input matrix with random values */
    srand((unsigned)time(NULL));
    for (int i = 0; i < width * height; ++i)
        h_in[i] = (float)rand() / RAND_MAX;

    /* Allocate device memory */
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc((void **)&d_in, size_in));
    CUDA_CHECK(cudaMalloc((void **)&d_out, size_out));

    /* Copy input to device */
    CUDA_CHECK(cudaMemcpy(d_in, h_in, size_in, cudaMemcpyHostToDevice));

    /* Kernel launch configuration */
    dim3 blockDim(TILE_DIM, BLOCK_ROWS, 1);
    dim3 gridDim((width  + TILE_DIM - 1) / TILE_DIM,
                 (height + TILE_DIM - 1) / TILE_DIM,
                 1);

    /* Timing events */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    /* Launch kernel */
    CUDA_CHECK(cudaEventRecord(start));
    transpose<<<gridDim, blockDim>>>(d_in, d_out, width, height);
    CUDA_CHECK(cudaEventRecord(stop));

    /* Synchronize and check for launch errors */
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Compute elapsed time */
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Transpose kernel execution time: %f ms\n", ms);

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_out, d_out, size_out, cudaMemcpyDeviceToHost));

    /* CPU reference */
    cpu_transpose(h_in, h_ref, width, height);

    /* Verify results */
    int mismatches = 0;
    for (int i = 0; i < width * height; ++i) {
        if (fabs(h_out[i] - h_ref[i]) > 1e-5f) {
            mismatches++;
            if (mismatches < 10) {
                printf("Mismatch at %d: GPU=%f, CPU=%f\n", i, h_out[i], h_ref[i]);
            }
        }
    }
    if (mismatches == 0)
        printf("Verification passed: all elements match.\n");
    else
        printf("Verification failed: %d mismatches found.\n", mismatches);

    /* Clean up */
    free(h_in);
    free(h_out);
    free(h_ref);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
```