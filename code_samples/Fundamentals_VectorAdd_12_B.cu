```c
/*
 * Aim: Matrix addition using CUDA, performing element-wise addition of two matrices A and B
 * to produce C, with boundary check to handle arbitrary matrix dimensions.
 *
 * Thinking:
 * - We need a CUDA kernel that will run in parallel on the GPU. Each thread will
 *   compute the sum for a single element of the resulting matrix.
 * - The grid of threads is organized into blocks, and each block contains a
 *   2D arrangement of threads. The global index of a thread in both dimensions
 *   is computed from the block index and the thread index.
 * - Since the matrix dimensions may not be an exact multiple of the block
 *   dimensions, some threads will map to coordinates outside the matrix.
 *   To avoid out‑of‑bounds memory accesses we add a boundary check:
 *     if (global_x < width && global_y < height)
 *   so that only threads that correspond to valid matrix elements perform
 *   the addition.
 * - The host code will:
 *     1. Allocate host memory for matrices A, B, and C.
 *     2. Initialize A and B with some values (for example, A[i][j] = i+j,
 *        B[i][j] = i-j).
 *     3. Allocate device memory and copy A and B to the device.
 *     4. Determine a suitable block and grid size (e.g., 16x16 threads per
 *        block).
 *     5. Launch the kernel.
 *     6. Copy the result back to the host and verify correctness.
 *     7. Free all allocated memory.
 *
 * The code below follows this plan. It includes error checking for CUDA
 * API calls and prints a success message when the computation is correct.
 */

#include <stdio.h>
#include <cuda_runtime.h>

// Kernel for element-wise matrix addition with boundary check
__global__ void matrixAdd(const float* A, const float* B, float* C,
                          int width, int height)
{
    // Compute global indices
    int global_x = blockIdx.x * blockDim.x + threadIdx.x;
    int global_y = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check to avoid out-of-bounds accesses
    if (global_x < width && global_y < height)
    {
        int idx = global_y * width + global_x;
        C[idx] = A[idx] + B[idx];
    }
}

// Helper macro for CUDA error checking
#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",            \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

int main()
{
    // Matrix dimensions
    const int width  = 1024;   // number of columns
    const int height = 768;    // number of rows
    const int size   = width * height;
    const int bytes  = size * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);

    // Initialize host matrices
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            int idx = y * width + x;
            h_A[idx] = (float)(x + y);
            h_B[idx] = (float)(x - y);
        }
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void**)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_B, bytes));
    CHECK_CUDA(cudaMalloc((void**)&d_C, bytes));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Define block and grid dimensions
    const dim3 blockSize(16, 16); // 16x16 threads per block
    const dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                       (height + blockSize.y - 1) / blockSize.y);

    // Launch kernel
    matrixAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, width, height);
    CHECK_CUDA(cudaGetLastError()); // Check for kernel launch errors

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify result
    int errors = 0;
    for (int y = 0; y < height && errors < 10; ++y)
    {
        for (int x = 0; x < width && errors < 10; ++x)
        {
            int idx = y * width + x;
            float expected = h_A[idx] + h_B[idx];
            if (fabs(h_C[idx] - expected) > 1e-5f)
            {
                printf("Mismatch at (%d, %d): GPU=%f, CPU=%f\n",
                       x, y, h_C[idx], expected);
                errors++;
            }
        }
    }

    if (errors == 0)
        printf("Matrix addition successful, no errors found.\n");
    else
        printf("Matrix addition found %d errors.\n", errors);

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));

    return 0;
}
```