```cuda
/*
The naive transpose reads from `input` uncoalesced and writes coalesced. This version reads coalesced and writes coalesced.

The goal of this program is to implement a CUDA kernel that performs a matrix transpose while ensuring that both the read from the source matrix and the write to the destination matrix are coalesced. In a naive transpose kernel, each thread would read an element from a non‑contiguous address (e.g., from different rows), which results in uncoalesced memory accesses. After loading the data, the write would be coalesced because threads in a warp write to contiguous locations in the transposed matrix. This version introduces a tiled transpose using shared memory so that each thread reads from a contiguous chunk of the input matrix and writes to a contiguous chunk of the output matrix. The tile is padded by one column to avoid shared‑memory bank conflicts. The kernel is launched with a 2D grid of thread blocks, each block handling a TILE_DIM × TILE_DIM tile. Each thread loads one element from the input into shared memory, performs a __syncthreads() to ensure all loads are complete, then each thread writes the transposed element from shared memory to global memory. Because the mapping of thread indices to global memory addresses aligns with memory strides, both the reads and writes are coalesced.

The host code allocates a random matrix, copies it to the GPU, launches the transpose kernel, copies the result back, and verifies correctness by comparing with a CPU implementation of the transpose. CUDA events are used to time the kernel execution. The program can be compiled with nvcc and run with optional command‑line arguments specifying the matrix width and height; if omitted, a 1024×1024 matrix is used.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_DIM 32
#define BLOCK_ROWS 8   // Can be tuned; BLOCK_ROWS divides TILE_DIM

// CUDA kernel performing a tiled transpose with coalesced reads and writes
__global__ void transposeCoalesced(float *out, const float *in, int width, int height)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 to avoid bank conflicts

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Read from global memory into shared memory (coalesced read)
    if (x < width && y < height)
    {
        tile[threadIdx.y][threadIdx.x] = in[y * width + x];
    }

    __syncthreads();

    // Write the transposed element from shared memory to global memory (coalesced write)
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    if (x < height && y < width)
    {
        out[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// CPU reference implementation of matrix transpose
void cpuTranspose(float *out, const float *in, int width, int height)
{
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            out[x * height + y] = in[y * width + x];
        }
    }
}

// Helper function to check CUDA errors
void checkCuda(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char **argv)
{
    int width  = 1024; // default width
    int height = 1024; // default height

    if (argc >= 3)
    {
        width  = atoi(argv[1]);
        height = atoi(argv[2]);
    }
    printf("Matrix size: %dx%d\n", width, height);

    size_t size = width * height * sizeof(float);

    // Allocate host memory
    float *h_in  = (float *)malloc(size);
    float *h_out = (float *)malloc(size);
    float *h_ref = (float *)malloc(size);
    if (!h_in || !h_out || !h_ref)
    {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // Initialize input matrix with random values
    for (int i = 0; i < width * height; ++i)
        h_in[i] = (float)rand() / RAND_MAX;

    // Allocate device memory
    float *d_in, *d_out;
    checkCuda(cudaMalloc((void **)&d_in,  size), "cudaMalloc d_in");
    checkCuda(cudaMalloc((void **)&d_out, size), "cudaMalloc d_out");

    // Copy input matrix to device
    checkCuda(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice), "cudaMemcpy HostToDevice");

    // Launch kernel
    dim3 blockDim(TILE_DIM, BLOCK_ROWS);
    dim3 gridDim((width  + TILE_DIM - 1) / TILE_DIM,
                 (height + TILE_DIM - 1) / TILE_DIM);

    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "cudaEventCreate start");
    checkCuda(cudaEventCreate(&stop),  "cudaEventCreate stop");

    checkCuda(cudaEventRecord(start, 0), "cudaEventRecord start");

    transposeCoalesced<<<gridDim, blockDim>>>(d_out, d_in, width, height);

    checkCuda(cudaEventRecord(stop, 0), "cudaEventRecord stop");
    checkCuda(cudaEventSynchronize(stop), "cudaEventSynchronize");

    float ms;
    checkCuda(cudaEventElapsedTime(&ms, start, stop), "cudaEventElapsedTime");
    printf("Kernel execution time: %f ms\n", ms);

    // Copy result back to host
    checkCuda(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost), "cudaMemcpy DeviceToHost");

    // Verify against CPU reference
    cpuTranspose(h_ref, h_in, width, height);

    bool ok = true;
    for (int i = 0; i < width * height; ++i)
    {
        if (fabs(h_out[i] - h_ref[i]) > 1e-5f)
        {
            ok = false;
            printf("Mismatch at index %d: GPU=%f, CPU=%f\n", i, h_out[i], h_ref[i]);
            break;
        }
    }

    if (ok)
        printf("Transpose verified: SUCCESS\n");
    else
        printf("Transpose verified: FAILURE\n");

    // Clean up
    free(h_in);
    free(h_out);
    free(h_ref);
    checkCuda(cudaFree(d_in),  "cudaFree d_in");
    checkCuda(cudaFree(d_out), "cudaFree d_out");
    checkCuda(cudaEventDestroy(start), "cudaEventDestroy start");
    checkCuda(cudaEventDestroy(stop),  "cudaEventDestroy stop");

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
```