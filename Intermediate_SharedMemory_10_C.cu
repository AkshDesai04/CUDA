```cuda
/* 
The naive transpose reads from `input` uncoalesced and writes coalesced. This version reads coalesced and writes coalesced. 

This CUDA program implements a tiled matrix transpose that ensures both reads from the global input matrix and writes to the global output matrix are performed in a coalesced manner. 
The kernel `transposeCoalesced` uses a shared memory tile to first gather elements from the input matrix in a row‑major (coalesced) access pattern. 
After loading the tile into shared memory, the threads then write the transposed elements back to global memory using a column‑major (coalesced) write pattern. 
The shared memory tile is padded to avoid shared memory bank conflicts. 

The host code allocates a square matrix of size N×N, initializes it, copies it to the GPU, launches the transpose kernel, copies the result back, and verifies correctness by comparing the transposed output to the expected result. 
The program also demonstrates simple performance measurement using CUDA events. 
The code is self‑contained and can be compiled with `nvcc -o transpose transpose.cu`. 
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8

// Utility macro for CUDA error checking
#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",        \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

// Kernel: Coalesced read and coalesced write transpose
__global__ void transposeCoalesced(const float *input, float *output, int width, int height)
{
    // Allocate shared memory tile with padding to avoid bank conflicts
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];

    // Global indices of the element to load
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Load element from global memory (coalesced if accessing row-major)
    if (x < width && y < height) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    } else {
        tile[threadIdx.y][threadIdx.x] = 0.0f; // Padding for out-of-range threads
    }

    __syncthreads();

    // Compute transposed coordinates
    int transposed_x = blockIdx.y * TILE_DIM + threadIdx.x;
    int transposed_y = blockIdx.x * TILE_DIM + threadIdx.y;

    // Write back to global memory (coalesced if accessing column-major)
    if (transposed_x < height && transposed_y < width) {
        output[transposed_y * height + transposed_x] = tile[threadIdx.x][threadIdx.y];
    }
}

// Helper function to print matrix (for small sizes)
void printMatrix(const float *mat, int width, int height)
{
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            printf("%6.1f ", mat[y * width + x]);
        }
        printf("\n");
    }
    printf("\n");
}

int main(void)
{
    // Matrix dimensions (must be multiples of TILE_DIM for simplicity)
    const int width  = 128; // columns of input
    const int height = 128; // rows of input
    const int N = width * height;

    // Allocate host memory
    float *h_input  = (float*)malloc(N * sizeof(float));
    float *h_output = (float*)malloc(N * sizeof(float));
    float *h_expected = (float*)malloc(N * sizeof(float));
    if (!h_input || !h_output || !h_expected) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input matrix with some pattern
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            h_input[y * width + x] = static_cast<float>(y * width + x);
        }
    }

    // Compute expected transpose on host
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            h_expected[x * height + y] = h_input[y * width + x];
        }
    }

    // Allocate device memory
    float *d_input = nullptr;
    float *d_output = nullptr;
    CHECK_CUDA(cudaMalloc((void**)&d_input, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_output, N * sizeof(float)));

    // Copy input to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, N * sizeof(float), cudaMemcpyHostToDevice));

    // Kernel launch parameters
    dim3 blockDim(TILE_DIM, BLOCK_ROWS, 1);
    dim3 gridDim((width  + TILE_DIM - 1) / TILE_DIM,
                 (height + TILE_DIM - 1) / TILE_DIM,
                 1);

    // Timing with CUDA events
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));

    // Launch kernel
    transposeCoalesced<<<gridDim, blockDim>>>(d_input, d_output, width, height);
    CHECK_CUDA(cudaGetLastError());

    // Stop timing
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    printf("Transpose kernel execution time: %.3f ms\n", ms);

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify correctness
    bool correct = true;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_output[i] - h_expected[i]) > 1e-5f) {
            correct = false;
            printf("Mismatch at index %d: host=%f, device=%f\n", i, h_expected[i], h_output[i]);
            break;
        }
    }
    if (correct) {
        printf("Transpose verified: SUCCESS\n");
    } else {
        printf("Transpose verified: FAILED\n");
    }

    // Optional: print matrices for small sizes
    if (width <= 8 && height <= 8) {
        printf("Input matrix:\n");
        printMatrix(h_input, width, height);
        printf("Transposed matrix (device):\n");
        printMatrix(h_output, height, width);
    }

    // Cleanup
    free(h_input);
    free(h_output);
    free(h_expected);
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return 0;
}
```