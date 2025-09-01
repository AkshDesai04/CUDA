```cuda
/*
Aim: Compare the performance of the shared memory version to a naive version where each thread reads all its required inputs directly from global memory.

Thinking:

1. To meaningfully compare shared memory usage versus naive global memory access, we need a kernel that requires each thread to read a small neighborhood of elements. A 2D convolution (3x3 filter) over an image is ideal because:
   - Each thread needs a 3x3 window of the input.
   - The naive implementation will perform nine global memory loads per thread.
   - The shared memory version can load a tile with halo once per block, then let all threads reuse these loads, dramatically reducing global memory traffic.

2. Image and block sizes:
   - Use a 1024×1024 grayscale image (1 byte per pixel). This is large enough to observe performance differences but still fits comfortably in GPU memory.
   - Use a 16×16 thread block size. Each block will load a (16+2)×(16+2) tile into shared memory to cover the halo region required for convolution.

3. Convolution kernel:
   - Use a simple 3×3 averaging filter (all weights 1/9) for clarity.
   - The filter is stored in constant memory on the device.

4. Kernels:
   - `naive_convolution`: Each thread reads its 3×3 neighborhood directly from global memory and writes the result to the output array.
   - `shared_convolution`: Each block loads its tile + halo into shared memory. After synchronization, each thread reads the 3×3 window from shared memory.

5. Timing:
   - Use CUDA events (`cudaEventCreate`, `cudaEventRecord`, `cudaEventElapsedTime`) to time each kernel execution accurately.
   - Run each kernel multiple times (e.g., 10 iterations) to average out timing fluctuations.

6. Validation:
   - After both kernels finish, copy the outputs back to host and compare pixel by pixel. Any mismatch indicates an implementation bug.

7. Output:
   - Print the average execution time for each kernel, the speedup factor, and whether the results match.

8. Error checking:
   - Wrap all CUDA API calls in a macro `CUDA_CHECK` that aborts on failure.

9. Self‑contained code:
   - All code (including data allocation, kernel definitions, timing, and cleanup) resides in a single `.cu` file. The code is ready to compile with `nvcc`.

The following code implements the above plan. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Macro for checking CUDA errors following a CUDA API call
#define CUDA_CHECK(call)                                                       \
    {                                                                          \
        cudaError_t err = call;                                                \
        if (err != cudaSuccess) {                                              \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",       \
                    __FILE__, __LINE__, cudaGetErrorString(err));              \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    }

// Image dimensions
#define IMG_WIDTH  1024
#define IMG_HEIGHT 1024

// Convolution filter dimensions
#define FILTER_SIZE 3
#define FILTER_RADIUS 1  // (FILTER_SIZE-1)/2

// Block dimensions
#define BLOCK_DIM 16

// Number of warm-up and timed iterations
#define WARMUP_ITERS 2
#define TIMED_ITERS 10

// Simple 3x3 averaging filter stored in constant memory
__constant__ float d_filter[FILTER_SIZE * FILTER_SIZE];

////////////////////////////////////////////////////////////////////////////////
// Naive convolution kernel: each thread loads its 3x3 neighborhood from global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void naive_convolution(const unsigned char *d_input,
                                  unsigned char *d_output,
                                  int width,
                                  int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // global x coordinate
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // global y coordinate

    // Handle boundary conditions: output zero for borders
    if (x < FILTER_RADIUS || y < FILTER_RADIUS ||
        x >= width - FILTER_RADIUS || y >= height - FILTER_RADIUS)
    {
        if (x < width && y < height)
            d_output[y * width + x] = 0;
        return;
    }

    float sum = 0.0f;
    for (int ky = -FILTER_RADIUS; ky <= FILTER_RADIUS; ++ky)
    {
        for (int kx = -FILTER_RADIUS; kx <= FILTER_RADIUS; ++kx)
        {
            int ix = x + kx;
            int iy = y + ky;
            float pixel = static_cast<float>(d_input[iy * width + ix]);
            float coeff = d_filter[(ky + FILTER_RADIUS) * FILTER_SIZE + (kx + FILTER_RADIUS)];
            sum += pixel * coeff;
        }
    }
    // Clamp to [0,255] and write to output
    d_output[y * width + x] = static_cast<unsigned char>(min(max(int(sum + 0.5f), 0), 255));
}

////////////////////////////////////////////////////////////////////////////////
// Shared memory convolution kernel: loads tile + halo into shared memory
////////////////////////////////////////////////////////////////////////////////
__global__ void shared_convolution(const unsigned char *d_input,
                                   unsigned char *d_output,
                                   int width,
                                   int height)
{
    // Shared memory tile size includes halo
    __shared__ unsigned char tile[BLOCK_DIM + 2 * FILTER_RADIUS][BLOCK_DIM + 2 * FILTER_RADIUS];

    int tx = threadIdx.x;  // thread's x index within block
    int ty = threadIdx.y;  // thread's y index within block

    int global_x = blockIdx.x * blockDim.x + tx;
    int global_y = blockIdx.y * blockDim.y + ty;

    // Compute indices in shared memory tile (offset by FILTER_RADIUS)
    int shared_x = tx + FILTER_RADIUS;
    int shared_y = ty + FILTER_RADIUS;

    // Load central tile element
    if (global_x < width && global_y < height)
        tile[shared_y][shared_x] = d_input[global_y * width + global_x];
    else
        tile[shared_y][shared_x] = 0;

    // Load halo regions
    // Left halo
    if (tx < FILTER_RADIUS)
    {
        int halo_x = global_x - FILTER_RADIUS;
        int halo_y = global_y;
        tile[shared_y][shared_x - FILTER_RADIUS] = (halo_x >= 0 && halo_y < height)
            ? d_input[halo_y * width + halo_x] : 0;
    }
    // Right halo
    if (tx >= blockDim.x - FILTER_RADIUS)
    {
        int halo_x = global_x + FILTER_RADIUS;
        int halo_y = global_y;
        tile[shared_y][shared_x + FILTER_RADIUS] = (halo_x < width && halo_y < height)
            ? d_input[halo_y * width + halo_x] : 0;
    }
    // Top halo
    if (ty < FILTER_RADIUS)
    {
        int halo_x = global_x;
        int halo_y = global_y - FILTER_RADIUS;
        tile[shared_y - FILTER_RADIUS][shared_x] = (halo_x < width && halo_y >= 0)
            ? d_input[halo_y * width + halo_x] : 0;
    }
    // Bottom halo
    if (ty >= blockDim.y - FILTER_RADIUS)
    {
        int halo_x = global_x;
        int halo_y = global_y + FILTER_RADIUS;
        tile[shared_y + FILTER_RADIUS][shared_x] = (halo_x < width && halo_y < height)
            ? d_input[halo_y * width + halo_x] : 0;
    }
    // Top-left corner halo
    if (tx < FILTER_RADIUS && ty < FILTER_RADIUS)
    {
        int halo_x = global_x - FILTER_RADIUS;
        int halo_y = global_y - FILTER_RADIUS;
        tile[shared_y - FILTER_RADIUS][shared_x - FILTER_RADIUS] =
            (halo_x >= 0 && halo_y >= 0)
            ? d_input[halo_y * width + halo_x] : 0;
    }
    // Top-right corner halo
    if (tx >= blockDim.x - FILTER_RADIUS && ty < FILTER_RADIUS)
    {
        int halo_x = global_x + FILTER_RADIUS;
        int halo_y = global_y - FILTER_RADIUS;
        tile[shared_y - FILTER_RADIUS][shared_x + FILTER_RADIUS] =
            (halo_x < width && halo_y >= 0)
            ? d_input[halo_y * width + halo_x] : 0;
    }
    // Bottom-left corner halo
    if (tx < FILTER_RADIUS && ty >= blockDim.y - FILTER_RADIUS)
    {
        int halo_x = global_x - FILTER_RADIUS;
        int halo_y = global_y + FILTER_RADIUS;
        tile[shared_y + FILTER_RADIUS][shared_x - FILTER_RADIUS] =
            (halo_x >= 0 && halo_y < height)
            ? d_input[halo_y * width + halo_x] : 0;
    }
    // Bottom-right corner halo
    if (tx >= blockDim.x - FILTER_RADIUS && ty >= blockDim.y - FILTER_RADIUS)
    {
        int halo_x = global_x + FILTER_RADIUS;
        int halo_y = global_y + FILTER_RADIUS;
        tile[shared_y + FILTER_RADIUS][shared_x + FILTER_RADIUS] =
            (halo_x < width && halo_y < height)
            ? d_input[halo_y * width + halo_x] : 0;
    }

    // Ensure all data is loaded into shared memory
    __syncthreads();

    // Handle boundary conditions
    if (global_x < FILTER_RADIUS || global_y < FILTER_RADIUS ||
        global_x >= width - FILTER_RADIUS || global_y >= height - FILTER_RADIUS)
    {
        if (global_x < width && global_y < height)
            d_output[global_y * width + global_x] = 0;
        return;
    }

    // Perform convolution using shared memory
    float sum = 0.0f;
    for (int ky = -FILTER_RADIUS; ky <= FILTER_RADIUS; ++ky)
    {
        for (int kx = -FILTER_RADIUS; kx <= FILTER_RADIUS; ++kx)
        {
            float pixel = static_cast<float>(
                tile[shared_y + ky][shared_x + kx]);
            float coeff = d_filter[(ky + FILTER_RADIUS) * FILTER_SIZE +
                                   (kx + FILTER_RADIUS)];
            sum += pixel * coeff;
        }
    }
    d_output[global_y * width + global_x] =
        static_cast<unsigned char>(min(max(int(sum + 0.5f), 0), 255));
}

////////////////////////////////////////////////////////////////////////////////
// Utility function to initialize the image with random data
////////////////////////////////////////////////////////////////////////////////
void initialize_image(unsigned char *img, int width, int height)
{
    for (int i = 0; i < width * height; ++i)
    {
        img[i] = rand() % 256;  // Random pixel value [0,255]
    }
}

////////////////////////////////////////////////////////////////////////////////
// Host main function
////////////////////////////////////////////////////////////////////////////////
int main()
{
    // Seed RNG
    srand(1234);

    // Allocate host image
    unsigned char *h_input  = (unsigned char *)malloc(IMG_WIDTH * IMG_HEIGHT);
    unsigned char *h_naive  = (unsigned char *)malloc(IMG_WIDTH * IMG_HEIGHT);
    unsigned char *h_shared = (unsigned char *)malloc(IMG_WIDTH * IMG_HEIGHT);

    if (!h_input || !h_naive || !h_shared)
    {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize input image with random values
    initialize_image(h_input, IMG_WIDTH, IMG_HEIGHT);

    // Device memory allocation
    unsigned char *d_input, *d_output_naive, *d_output_shared;
    CUDA_CHECK(cudaMalloc((void **)&d_input,          IMG_WIDTH * IMG_HEIGHT));
    CUDA_CHECK(cudaMalloc((void **)&d_output_naive,   IMG_WIDTH * IMG_HEIGHT));
    CUDA_CHECK(cudaMalloc((void **)&d_output_shared,  IMG_WIDTH * IMG_HEIGHT));

    // Copy input image to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input,
                          IMG_WIDTH * IMG_HEIGHT,
                          cudaMemcpyHostToDevice));

    // Define filter (3x3 averaging)
    float h_filter[FILTER_SIZE * FILTER_SIZE] = {
        1.0f/9, 1.0f/9, 1.0f/9,
        1.0f/9, 1.0f/9, 1.0f/9,
        1.0f/9, 1.0f/9, 1.0f/9
    };
    CUDA_CHECK(cudaMemcpyToSymbol(d_filter, h_filter,
                                  FILTER_SIZE * FILTER_SIZE * sizeof(float),
                                  0,
                                  cudaMemcpyHostToDevice));

    // Kernel launch configuration
    dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
    dim3 gridDim((IMG_WIDTH + BLOCK_DIM - 1) / BLOCK_DIM,
                 (IMG_HEIGHT + BLOCK_DIM - 1) / BLOCK_DIM);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float naive_time = 0.0f;
    float shared_time = 0.0f;

    // Warm-up and timed iterations for naive kernel
    for (int i = 0; i < WARMUP_ITERS + TIMED_ITERS; ++i)
    {
        CUDA_CHECK(cudaEventRecord(start));
        naive_convolution<<<gridDim, blockDim>>>(d_input, d_output_naive,
                                                 IMG_WIDTH, IMG_HEIGHT);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        if (i >= WARMUP_ITERS)
        {
            float ms;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            naive_time += ms;
        }
    }
    naive_time /= TIMED_ITERS;

    // Warm-up and timed iterations for shared memory kernel
    for (int i = 0; i < WARMUP_ITERS + TIMED_ITERS; ++i)
    {
        CUDA_CHECK(cudaEventRecord(start));
        shared_convolution<<<gridDim, blockDim>>>(d_input, d_output_shared,
                                                  IMG_WIDTH, IMG_HEIGHT);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        if (i >= WARMUP_ITERS)
        {
            float ms;
            CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
            shared_time += ms;
        }
    }
    shared_time /= TIMED_ITERS;

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_naive,  d_output_naive,
                          IMG_WIDTH * IMG_HEIGHT,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_shared, d_output_shared,
                          IMG_WIDTH * IMG_HEIGHT,
                          cudaMemcpyDeviceToHost));

    // Validate results
    int mismatches = 0;
    for (int i = 0; i < IMG_WIDTH * IMG_HEIGHT; ++i)
    {
        if (h_naive[i] != h_shared[i])
            ++mismatches;
    }

    // Output timing results
    printf("Average execution time over %d iterations:\n", TIMED_ITERS);
    printf("  Naive kernel      : %f ms\n", naive_time);
    printf("  Shared memory kernel: %f ms\n", shared_time);
    printf("Speedup: %.2fx\n", naive_time / shared_time);
    if (mismatches == 0)
        printf("Validation PASSED: Outputs match.\n");
    else
        printf("Validation FAILED: %d mismatched pixels.\n", mismatches);

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output_naive));
    CUDA_CHECK(cudaFree(d_output_shared));
    free(h_input);
    free(h_naive);
    free(h_shared);

    return EXIT_SUCCESS;
}
```