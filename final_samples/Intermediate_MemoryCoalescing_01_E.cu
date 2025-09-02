```cuda
/* 
Aim of the program: Profile this corrected naive transpose kernel

Thinking:
To measure the performance of a simple (naive) matrix transpose kernel, we need
to set up a small CUDA program that:
1. Generates a square matrix on the host, initializes it with known data.
2. Allocates device memory for the input and output matrices.
3. Copies the input matrix from host to device.
4. Launches the naive transpose kernel, timing its execution with CUDA
   events (cudaEventRecord and cudaEventElapsedTime).
5. Copies the transposed matrix back to the host.
6. Verifies that the transpose was performed correctly by comparing the
   result with a CPU reference.
7. Prints the elapsed kernel time in milliseconds.

The kernel itself is deliberately naive: each thread reads one element from
the source matrix and writes it to the transposed position in the destination
matrix. No shared memory tiling or coalescing optimizations are used, so
this will provide a baseline for profiling.

We will use a block size of 32x32 threads (the typical warp size aligned
block). The grid dimensions are computed to cover the entire matrix. For
simplicity, the matrix is square (N x N) and we hard‑code N to 1024, but
this can be changed easily.

The program includes basic error checking macros to catch CUDA API errors.
All output is limited to the kernel execution time and a success/failure
message for the transpose verification.

Compile with:
    nvcc -O2 -o transpose_profile transpose_profile.cu
Run:
    ./transpose_profile
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Macro for checking CUDA errors following a CUDA API call or kernel launch
#define CHECK_CUDA(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",               \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// Naive transpose kernel
__global__ void transpose_naive(int *odata, const int *idata, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;  // column index in output
    int y = blockIdx.y * blockDim.y + threadIdx.y;  // row index in output

    if (x < width && y < height)
    {
        // source index: y (row) * width + x (col)
        // destination index: x (row) * height + y (col)
        odata[x * height + y] = idata[y * width + x];
    }
}

int main(void)
{
    const int N = 1024;           // Matrix dimension (NxN)
    const size_t bytes = N * N * sizeof(int);

    // Allocate host memory
    int *h_A = (int *)malloc(bytes);
    int *h_B = (int *)malloc(bytes);
    int *h_ref = (int *)malloc(bytes);
    if (!h_A || !h_B || !h_ref)
    {
        fprintf(stderr, "Failed to allocate host matrices\n");
        return EXIT_FAILURE;
    }

    // Initialize host matrix A with some values (e.g., row-major indices)
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            h_A[i * N + j] = i * N + j;
        }
    }

    // Compute reference transpose on host for verification
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            h_ref[j * N + i] = h_A[i * N + j];
        }
    }

    // Allocate device memory
    int *d_A = NULL;
    int *d_B = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_B, bytes));

    // Copy input matrix to device
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));

    // Define block and grid sizes
    dim3 blockSize(32, 32);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (N + blockSize.y - 1) / blockSize.y);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    // Record start event
    CHECK_CUDA(cudaEventRecord(start, 0));

    // Launch kernel
    transpose_naive<<<gridSize, blockSize>>>(d_B, d_A, N, N);
    CHECK_CUDA(cudaGetLastError()); // Check kernel launch

    // Record stop event
    CHECK_CUDA(cudaEventRecord(stop, 0));

    // Wait for the stop event to complete
    CHECK_CUDA(cudaEventSynchronize(stop));

    // Compute elapsed time
    float milliseconds = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_B, d_B, bytes, cudaMemcpyDeviceToHost));

    // Verify correctness
    int mismatches = 0;
    for (int i = 0; i < N * N; ++i)
    {
        if (h_B[i] != h_ref[i])
        {
            mismatches++;
            if (mismatches <= 10)
                fprintf(stderr, "Mismatch at index %d: got %d, expected %d\n",
                        i, h_B[i], h_ref[i]);
        }
    }

    if (mismatches == 0)
    {
        printf("Transpose verified successfully.\n");
    }
    else
    {
        printf("Transpose verification failed with %d mismatches.\n", mismatches);
    }

    // Print kernel execution time
    printf("Naive transpose kernel execution time: %.3f ms\n", milliseconds);

    // Cleanup
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    free(h_A);
    free(h_B);
    free(h_ref);

    return EXIT_SUCCESS;
}
```