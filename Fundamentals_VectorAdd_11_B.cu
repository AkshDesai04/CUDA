/*
Aim of the program (word for word):
Launch your grid-stride kernel for a vector of size 1M, but only launch a grid of 1024 threads total. Verify the correctness.

Thinking:
To satisfy the requirement of exactly 1024 total threads, the simplest choice is to launch a single block of 1024 threads
(blockDim.x = 1024, gridDim.x = 1).  This gives us a total of 1024 threads.  The kernel will use a grid‑stride loop to
iterate over a 1‑million element vector.  With a stride equal to blockDim.x * gridDim.x = 1024, each thread will process
roughly 1M / 1024 = 1024 elements.  We choose a simple operation (adding 1.0f) so that we can verify correctness easily
by checking that every element ends up equal to 1.0f after the kernel execution.

Steps in the program:
1. Define N = 1<<20 (1M).
2. Allocate host and device arrays of float.
3. Initialize host array to zeros.
4. Copy to device.
5. Launch kernel with <<<1, 1024>>>.
6. Synchronize and copy result back.
7. Verify each element equals 1.0f; report success or failure.
8. Clean up resources.

We also include basic CUDA error checking to catch any runtime errors.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N (1 << 20)           // 1M elements
#define BLOCK_SIZE 1024       // Total threads = BLOCK_SIZE * GRID_SIZE = 1024
#define GRID_SIZE 1           // Single block

__global__ void gridStrideAdd(float *a, int n)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < n; i += stride)
    {
        a[i] += 1.0f;
    }
}

int main(void)
{
    float *h_a = NULL;
    float *d_a = NULL;
    size_t size = N * sizeof(float);

    // Allocate host memory
    h_a = (float *)malloc(size);
    if (h_a == NULL)
    {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host array to 0.0f
    for (int i = 0; i < N; ++i)
    {
        h_a[i] = 0.0f;
    }

    // Allocate device memory
    cudaError_t err = cudaMalloc((void **)&d_a, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        free(h_a);
        return EXIT_FAILURE;
    }

    // Copy data from host to device
    err = cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_a);
        free(h_a);
        return EXIT_FAILURE;
    }

    // Launch kernel
    gridStrideAdd<<<GRID_SIZE, BLOCK_SIZE>>>(d_a, N);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_a);
        free(h_a);
        return EXIT_FAILURE;
    }

    // Synchronize
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_a);
        free(h_a);
        return EXIT_FAILURE;
    }

    // Copy result back to host
    err = cudaMemcpy(h_a, d_a, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_a);
        free(h_a);
        return EXIT_FAILURE;
    }

    // Verify correctness
    int errors = 0;
    for (int i = 0; i < N; ++i)
    {
        if (h_a[i] != 1.0f)
        {
            if (errors < 10) // Print at most 10 error locations
            {
                fprintf(stderr, "Error at index %d: %f\n", i, h_a[i]);
            }
            errors++;
        }
    }

    if (errors == 0)
    {
        printf("Verification passed! All elements are 1.0f.\n");
    }
    else
    {
        printf("Verification failed! %d mismatches found.\n", errors);
    }

    // Clean up
    cudaFree(d_a);
    free(h_a);

    return (errors == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
