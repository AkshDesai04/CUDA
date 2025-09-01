```cuda
/*
Implement a matrix transpose: `output[x * height + y] = input[y * width + x]`. Note this can have memory coalescing issues.

Thinking:
The goal is to provide a CUDA program that performs a matrix transpose. The straightforward way is to launch a kernel where each thread handles one element of the output matrix, computing its position from the input coordinates. This naive approach leads to uncoalesced memory accesses because threads in the same warp read consecutive elements from the input array that are not contiguous in memory (the input is stored in row-major order). Similarly, the writes to the output may also be uncoalesced if the output is also in row-major order. 

To keep the example simple while still illustrating the problem, I will implement:
1. A naive transpose kernel (`transpose_naive`) that uses one thread per element and directly applies the formula.
2. A main function that:
   - Creates a sample matrix on the host.
   - Allocates device memory.
   - Copies the matrix to the device.
   - Launches the kernel with a 2D grid of thread blocks.
   - Copies the result back to the host.
   - Validates that the transpose was performed correctly.
   - Prints some elements for sanity checking.
   - Cleans up memory.

Optionally, I also mention that a tiled transpose (using shared memory) would resolve the coalescing issue, but it is not included here to keep the code concise.

The program is self‑contained and can be compiled with `nvcc -o transpose transpose.cu` and executed. The output will show a few elements of the transposed matrix to confirm correctness.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// Naive transpose kernel: each thread writes one element of the output
__global__ void transpose_naive(const float *in, float *out, int width, int height)
{
    // Compute the (x, y) coordinate of the element this thread will process
    int x = blockIdx.x * blockDim.x + threadIdx.x; // column index in output
    int y = blockIdx.y * blockDim.y + threadIdx.y; // row index in output

    if (x < width && y < height)
    {
        // The formula from the prompt
        // Note: input is stored as input[y * width + x] (row-major)
        // We compute output[x * height + y] (column-major)
        out[x * height + y] = in[y * width + x];
    }
}

// Utility function to check CUDA errors
void checkCuda(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main()
{
    // Define matrix dimensions
    const int width = 5;   // columns
    const int height = 4;  // rows
    const int N = width * height;

    // Allocate host memory and initialize with some values
    float *h_in = (float*)malloc(N * sizeof(float));
    float *h_out = (float*)malloc(N * sizeof(float));
    if (!h_in || !h_out)
    {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Fill input matrix with a simple pattern: element = row * width + col
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            h_in[y * width + x] = static_cast<float>(y * width + x);
        }
    }

    // Allocate device memory
    float *d_in = nullptr;
    float *d_out = nullptr;
    checkCuda(cudaMalloc((void**)&d_in, N * sizeof(float)), "cudaMalloc d_in");
    checkCuda(cudaMalloc((void**)&d_out, N * sizeof(float)), "cudaMalloc d_out");

    // Copy input data to device
    checkCuda(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy H2D");

    // Define block and grid sizes
    const int BLOCK_SIZE = 16;
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((width  + BLOCK_SIZE - 1) / BLOCK_SIZE,
                 (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch the naive transpose kernel
    transpose_naive<<<gridDim, blockDim>>>(d_in, d_out, width, height);
    checkCuda(cudaGetLastError(), "Kernel launch");

    // Copy result back to host
    checkCuda(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy D2H");

    // Validate the result
    bool valid = true;
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            float expected = h_in[y * width + x];
            float actual = h_out[x * height + y];
            if (expected != actual)
            {
                fprintf(stderr, "Mismatch at (%d,%d): expected %f, got %f\n",
                        y, x, expected, actual);
                valid = false;
                break;
            }
        }
        if (!valid) break;
    }

    if (valid)
        printf("Transpose successful. Sample output:\n");
    else
        printf("Transpose failed.\n");

    // Print a few elements of the transposed matrix
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            printf("%6.1f ", h_out[x * height + y]);
        }
        printf("\n");
    }

    // Clean up
    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);

    return 0;
}
```