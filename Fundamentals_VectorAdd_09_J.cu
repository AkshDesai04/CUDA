/*
Aim of the program: Implement `D[i] = sqrt(A[i]*A[i] + B[i]*B[i])` (vector magnitude for 2D vectors).

My thinking process for this task was as follows:

1. The goal is to compute the magnitude of 2‑D vectors for a large array of points.  
   The mathematical formula for the magnitude of a vector (A[i], B[i]) is  
   `sqrt(A[i]^2 + B[i]^2)`.  
   In CUDA, this computation is embarrassingly parallel – each element can be computed independently, so a simple element‑wise kernel is appropriate.

2. I decided to write a small self‑contained CUDA C program that:
   - allocates three large arrays (A, B, D) on the host,
   - initializes A and B with some test data,
   - copies them to device memory,
   - launches a kernel where each thread computes one element of D,
   - copies the result back,
   - validates a few entries, and
   - cleans up.

3. For the kernel, I used `sqrtf` for single‑precision floats.  
   The kernel takes pointers to device arrays and an integer `N` for the number of elements.  
   Each thread calculates its global index `idx = blockIdx.x * blockDim.x + threadIdx.x` and, if `idx < N`, performs the calculation.

4. I chose a simple block size of 256 threads and calculated the number of blocks as `(N + blockSize - 1) / blockSize`.

5. Basic error checking after CUDA API calls is added to catch potential failures.

6. To keep the program straightforward, I used `float` arrays and a `#define` for `N` to make it easy to change the array size.

7. The program prints the first 10 computed magnitudes to verify correctness and prints a message when the computation is finished.

Now the complete CUDA code follows. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define N (1 << 20)          // Number of elements (1M)
#define BLOCK_SIZE 256       // Threads per block

// Kernel to compute vector magnitudes
__global__ void vectorMagnitude(const float *A, const float *B, float *D, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float a = A[idx];
        float b = B[idx];
        D[idx] = sqrtf(a * a + b * b);
    }
}

// Helper function for CUDA error checking
void checkCuda(cudaError_t err, const char *msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s : %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main() {
    // Host allocations
    float *h_A = (float *)malloc(N * sizeof(float));
    float *h_B = (float *)malloc(N * sizeof(float));
    float *h_D = (float *)malloc(N * sizeof(float));
    if (!h_A || !h_B || !h_D) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays with sample data
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;               // Example data
        h_B[i] = (float)(i * 2);         // Example data
    }

    // Device allocations
    float *d_A = NULL, *d_B = NULL, *d_D = NULL;
    checkCuda(cudaMalloc((void **)&d_A, N * sizeof(float)), "cudaMalloc d_A");
    checkCuda(cudaMalloc((void **)&d_B, N * sizeof(float)), "cudaMalloc d_B");
    checkCuda(cudaMalloc((void **)&d_D, N * sizeof(float)), "cudaMalloc d_D");

    // Copy data from host to device
    checkCuda(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy A");
    checkCuda(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice), "cudaMemcpy B");

    // Kernel launch parameters
    int gridSize = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Launch the kernel
    vectorMagnitude<<<gridSize, BLOCK_SIZE>>>(d_A, d_B, d_D, N);
    checkCuda(cudaGetLastError(), "Kernel launch");

    // Wait for GPU to finish before accessing on host
    checkCuda(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    // Copy result back to host
    checkCuda(cudaMemcpy(h_D, d_D, N * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy D");

    // Simple validation: print first 10 results
    printf("First 10 computed magnitudes:\n");
    for (int i = 0; i < 10; ++i) {
        printf("D[%d] = %f\n", i, h_D[i]);
    }

    // Clean up
    free(h_A);
    free(h_B);
    free(h_D);
    checkCuda(cudaFree(d_A), "cudaFree d_A");
    checkCuda(cudaFree(d_B), "cudaFree d_B");
    checkCuda(cudaFree(d_D), "cudaFree d_D");

    printf("Computation finished successfully.\n");
    return EXIT_SUCCESS;
}
