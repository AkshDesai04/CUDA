/*
Write a kernel to set a matrix to the identity matrix (1s on the diagonal, 0s elsewhere).

The aim of this program is to demonstrate a simple CUDA kernel that transforms a square matrix
into its identity form. The matrix is stored in row-major order and represented as a 1D array.
Each thread in the kernel is responsible for one element of the matrix. It checks whether
its assigned position lies on the main diagonal; if so, it writes a value of 1.0, otherwise
0.0. The kernel is launched with a 2D grid of thread blocks to cover all rows and columns.

In addition to the kernel, the host code allocates the matrix on both host and device,
initializes it to zero (for clarity), copies it to the device, launches the kernel, copies
back the result, and prints the matrix to confirm that the identity has been formed.

The code includes basic error checking using cudaGetLastError and checks the return values
of CUDA API calls. The matrix dimension N is defined as a compile‑time constant, but it
can easily be changed or made dynamic.

This example is self‑contained and can be compiled with nvcc:
    nvcc -o identity identity.cu
and run:
    ./identity
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 8          // Matrix dimension (N x N)
#define THREADS_PER_BLOCK 16

// CUDA kernel to set a matrix to the identity matrix
__global__ void set_identity_kernel(float *matrix, int width, int height)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y; // Row index
    int col = blockIdx.x * blockDim.x + threadIdx.x; // Column index

    if (row < height && col < width)
    {
        // Compute linear index in row-major order
        int idx = row * width + col;
        matrix[idx] = (row == col) ? 1.0f : 0.0f;
    }
}

// Helper function to print the matrix
void print_matrix(const float *matrix, int width, int height)
{
    for (int i = 0; i < height; ++i)
    {
        for (int j = 0; j < width; ++j)
        {
            printf("%6.1f ", matrix[i * width + j]);
        }
        printf("\n");
    }
}

int main(void)
{
    // Host allocation
    size_t size = N * N * sizeof(float);
    float *h_matrix = (float *)malloc(size);
    if (h_matrix == NULL)
    {
        fprintf(stderr, "Failed to allocate host matrix\n");
        return EXIT_FAILURE;
    }

    // Initialize host matrix to zeros (optional, just for clarity)
    for (int i = 0; i < N * N; ++i)
    {
        h_matrix[i] = 0.0f;
    }

    // Device allocation
    float *d_matrix = NULL;
    cudaError_t err = cudaMalloc((void **)&d_matrix, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        free(h_matrix);
        return EXIT_FAILURE;
    }

    // Copy host matrix to device (not strictly necessary since kernel writes all entries)
    err = cudaMemcpy(d_matrix, h_matrix, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_matrix);
        free(h_matrix);
        return EXIT_FAILURE;
    }

    // Define grid and block dimensions
    dim3 threadsPerBlock(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch kernel
    set_identity_kernel<<<numBlocks, threadsPerBlock>>>(d_matrix, N, N);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_matrix);
        free(h_matrix);
        return EXIT_FAILURE;
    }

    // Wait for device to finish
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_matrix);
        free(h_matrix);
        return EXIT_FAILURE;
    }

    // Copy result back to host
    err = cudaMemcpy(h_matrix, d_matrix, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "cudaMemcpy D2H failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_matrix);
        free(h_matrix);
        return EXIT_FAILURE;
    }

    // Print the resulting matrix
    printf("Identity matrix (%d x %d):\n", N, N);
    print_matrix(h_matrix, N, N);

    // Clean up
    cudaFree(d_matrix);
    free(h_matrix);

    return EXIT_SUCCESS;
}
