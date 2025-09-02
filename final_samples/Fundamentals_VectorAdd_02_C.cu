/*
Wrap the grid/block calculation and kernel launch into a helper function.
Thinking:
The goal is to demonstrate how to encapsulate the logic that computes the
number of threads per block and blocks per grid, and then performs the
kernel launch, into a reusable helper function.  A simple vector addition
kernel is used as an example: C = A + B.  The helper function takes the
problem size and device pointers, calculates an appropriate grid and
block configuration (using a typical block size of 256 threads), and
launches the kernel.  This keeps the main program clean and highlights
how to isolate launch details for better readability and potential reuse
in larger applications.  Basic error checking is performed via a macro
to ensure that CUDA API calls are validated.  The program allocates
memory on the host and device, initializes input vectors, invokes the
helper, and finally verifies the result.
*/

#include <stdio.h>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call)                                              \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

// Kernel: vector addition
__global__ void vecAdd(const float* A, const float* B, float* C, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        C[idx] = A[idx] + B[idx];
}

// Helper function that calculates grid/block dims and launches the kernel
void launchVecAddKernel(int N, const float* d_A, const float* d_B, float* d_C)
{
    // Choose a reasonable block size
    const int threadsPerBlock = 256;
    // Compute number of blocks needed to cover all elements
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the kernel
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Check for launch errors
    CUDA_CHECK(cudaGetLastError());
    // Wait for GPU to finish before returning
    CUDA_CHECK(cudaDeviceSynchronize());
}

int main()
{
    const int N = 1 << 20; // Example: 1M elements
    size_t size = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(2 * i);
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size));

    // Copy host data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Launch kernel via helper
    launchVecAddKernel(N, d_A, d_B, d_C);

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Verify a few elements
    bool success = true;
    for (int i = 0; i < 10; ++i) {
        float expected = h_A[i] + h_B[i];
        if (h_C[i] != expected) {
            printf("Mismatch at index %d: expected %f, got %f\n", i, expected, h_C[i]);
            success = false;
            break;
        }
    }
    if (success) {
        printf("Vector addition successful. Sample values:\n");
        for (int i = 0; i < 5; ++i) {
            printf("C[%d] = %f\n", i, h_C[i]);
        }
    }

    // Clean up
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    return 0;
}
