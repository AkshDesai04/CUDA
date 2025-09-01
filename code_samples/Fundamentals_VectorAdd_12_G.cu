/*
Aim of the program:
Launch with a 1D grid of 1D blocks instead. Each thread must manually compute its (x,y) coordinate from its global 1D ID: int x = id % width; int y = id / width;.

Thought process:
1. We need to create a simple CUDA program that demonstrates launching a kernel with a 1D grid and 1D blocks.
2. Each thread will calculate its unique global linear index (`id`) by combining block and thread indices: `id = blockIdx.x * blockDim.x + threadIdx.x`.
3. From this linear index, we will compute the 2D coordinates `(x, y)` using the given formulas:
   ```
   int x = id % width;
   int y = id / width;
   ```
   where `width` is the number of columns of the output array.
4. The kernel will store a value derived from `(x, y)` into a linear array that represents a 2D grid in row-major order. For demonstration, we'll store the sum `x + y`.
5. The host code will allocate memory on both host and device, initialize the parameters, launch the kernel with a suitable number of threads per block and number of blocks to cover `width * height` elements.
6. After kernel execution, we will copy the result back to the host and print a small portion of the array to verify correctness.
7. The code includes error checking for CUDA API calls, which is a good practice.
8. All code is encapsulated in a single .cu file, with the required multiline comment at the top containing both the aim statement and the above explanation.

*/

#include <stdio.h>
#include <cuda_runtime.h>

// Macro for checking CUDA errors
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",            \
                    __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

// Kernel that writes a simple value derived from (x, y) coordinates
__global__ void compute_xy(int *out, int width, int height)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;

    if (id >= total) return; // guard against overrun

    int x = id % width;
    int y = id / width;

    // Example computation: store the sum of coordinates
    out[id] = x + y;
}

int main(void)
{
    const int width  = 8;   // number of columns
    const int height = 6;   // number of rows
    const int N = width * height;

    // Host allocation
    int *h_out = (int *)malloc(N * sizeof(int));
    if (!h_out) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Device allocation
    int *d_out;
    CUDA_CHECK(cudaMalloc((void **)&d_out, N * sizeof(int)));

    // Kernel launch parameters: 1D grid, 1D blocks
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch kernel
    compute_xy<<<blocks, threadsPerBlock>>>(d_out, width, height);

    // Check for launch errors
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print a simple representation of the 2D array
    printf("Resulting 2D array (x + y):\n");
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            printf("%2d ", h_out[y * width + x]);
        }
        printf("\n");
    }

    // Clean up
    free(h_out);
    CUDA_CHECK(cudaFree(d_out));

    return EXIT_SUCCESS;
}
