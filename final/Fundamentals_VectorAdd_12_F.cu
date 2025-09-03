```cuda
/*
Aim of the program: Implement a matrix transpose: `output[x * height + y] = input[y * width + x]`. Note this can have memory coalescing issues.

Thinking:
The goal is to provide a CUDA implementation of a matrix transpose that follows the given equation.  
The program contains:
1. A naive kernel that performs the transpose directly. This kernel will suffer from memory coalescing problems because the reads from the source array (`input`) are not coalesced when `x` varies across threads in a warp, and writes to the destination (`output`) are also not coalesced for the same reason.
2. An optional shared‑memory tiled kernel that improves performance by using shared memory and avoiding bank conflicts.
3. Host code that:
   - Parses command line arguments for matrix width and height.
   - Allocates host memory and initializes the input matrix with random values.
   - Allocates device memory, copies the input to the device.
   - Launches the transpose kernel with an appropriate grid and block configuration.
   - Copies the transposed matrix back to the host.
   - Performs a simple verification for a few elements to ensure correctness.
   - Measures execution time using CUDA events.

The code is self‑contained and can be compiled with `nvcc`. It demonstrates how to handle memory coalescing issues in a transpose operation and how to mitigate them using shared memory.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <ctime>

// Naive transpose kernel (subject to memory coalescing issues)
__global__ void transposeNaive(float *o, const float *i, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; // column index in input
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row index in input

    if (x < width && y < height)
    {
        // Linear index of input: row-major
        int inIdx = y * width + x;
        // Linear index of output: column-major (since output stride is height)
        int outIdx = x * height + y;
        o[outIdx] = i[inIdx];
    }
}

// Optimized transpose kernel using shared memory (tiled, no bank conflicts)
#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transposeShared(float *o, const float *i, int width, int height)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 to avoid bank conflicts

    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
    int index_in = yIndex * width + xIndex;

    // Load data from global memory to shared memory
    if (xIndex < width && yIndex < height)
    {
        tile[threadIdx.y][threadIdx.x] = i[index_in];
    }
    __syncthreads();

    // Write the transposed data to global memory
    int transposed_x = blockIdx.y * TILE_DIM + threadIdx.x;
    int transposed_y = blockIdx.x * TILE_DIM + threadIdx.y;
    int index_out = transposed_y * height + transposed_x; // output stride is height

    if (transposed_x < height && transposed_y < width)
    {
        o[index_out] = tile[threadIdx.x][threadIdx.y];
    }
}

// Helper function for error checking
void checkCuda(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA error (%s): %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(int argc, char *argv[])
{
    if (argc != 3)
    {
        fprintf(stderr, "Usage: %s <width> <height>\n", argv[0]);
        return EXIT_FAILURE;
    }

    int width = atoi(argv[1]);   // number of columns in input
    int height = atoi(argv[2]);  // number of rows in input

    size_t size = (size_t)width * height * sizeof(float);

    // Allocate host memory
    float *h_in  = (float*)malloc(size);
    float *h_out = (float*)malloc(size);
    if (!h_in || !h_out)
    {
        fprintf(stderr, "Host memory allocation failed\n");
        return EXIT_FAILURE;
    }

    // Initialize input matrix with random values
    srand((unsigned)time(NULL));
    for (size_t i = 0; i < (size_t)width * height; ++i)
        h_in[i] = static_cast<float>(rand()) / RAND_MAX;

    // Allocate device memory
    float *d_in = nullptr, *d_out = nullptr;
    checkCuda(cudaMalloc((void**)&d_in,  size), "malloc d_in");
    checkCuda(cudaMalloc((void**)&d_out, size), "malloc d_out");

    // Copy input to device
    checkCuda(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice), "Memcpy H2D");

    // Define block and grid dimensions
    dim3 blockDim(32, 8);
    dim3 gridDim((width  + blockDim.x - 1) / blockDim.x,
                 (height + blockDim.y - 1) / blockDim.y);

    // Timing events
    cudaEvent_t start, stop;
    checkCuda(cudaEventCreate(&start), "event create start");
    checkCuda(cudaEventCreate(&stop),  "event create stop");

    // Launch naive transpose kernel
    checkCuda(cudaEventRecord(start, 0), "event record start");
    transposeNaive<<<gridDim, blockDim>>>(d_out, d_in, width, height);
    checkCuda(cudaEventRecord(stop, 0),  "event record stop");
    checkCuda(cudaEventSynchronize(stop), "event synchronize");

    float milliseconds = 0;
    checkCuda(cudaEventElapsedTime(&milliseconds, start, stop), "event elapsed time");
    printf("Naive transpose kernel time: %f ms\n", milliseconds);

    // Copy result back to host
    checkCuda(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost), "Memcpy D2H");

    // Simple correctness check (compare a few elements)
    bool correct = true;
    for (int i = 0; i < 5; ++i)
    {
        int x = rand() % width;
        int y = rand() % height;
        float expected = h_in[y * width + x];
        float got = h_out[x * height + y];
        if (fabs(expected - got) > 1e-5f)
        {
            fprintf(stderr, "Mismatch at (x=%d, y=%d): expected %f, got %f\n",
                    x, y, expected, got);
            correct = false;
            break;
        }
    }
    printf("Naive transpose correctness: %s\n", correct ? "PASS" : "FAIL");

    // Optional: run optimized shared-memory transpose
    // Reset timing
    checkCuda(cudaEventRecord(start, 0), "event record start");
    transposeShared<<<gridDim, blockDim>>>(d_out, d_in, width, height);
    checkCuda(cudaEventRecord(stop, 0),  "event record stop");
    checkCuda(cudaEventSynchronize(stop), "event synchronize");

    checkCuda(cudaEventElapsedTime(&milliseconds, start, stop), "event elapsed time");
    printf("Shared-memory transpose kernel time: %f ms\n", milliseconds);

    // Copy back and verify again
    checkCuda(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost), "Memcpy D2H");
    correct = true;
    for (int i = 0; i < 5; ++i)
    {
        int x = rand() % width;
        int y = rand() % height;
        float expected = h_in[y * width + x];
        float got = h_out[x * height + y];
        if (fabs(expected - got) > 1e-5f)
        {
            fprintf(stderr, "Mismatch at (x=%d, y=%d): expected %f, got %f\n",
                    x, y, expected, got);
            correct = false;
            break;
        }
    }
    printf("Shared-memory transpose correctness: %s\n", correct ? "PASS" : "FAIL");

    // Clean up
    checkCuda(cudaFree(d_in), "free d_in");
    checkCuda(cudaFree(d_out), "free d_out");
    free(h_in);
    free(h_out);
    checkCuda(cudaEventDestroy(start), "event destroy start");
    checkCuda(cudaEventDestroy(stop),  "event destroy stop");

    return EXIT_SUCCESS;
}
```