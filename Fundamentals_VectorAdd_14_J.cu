```cuda
/*
Aim of the program:
Handle large `double` vectors with boundary checks.

Thought process:
- The goal is to demonstrate how to process very large arrays of type `double` on the GPU while ensuring that each thread accesses only valid memory addresses.
- The program performs element‑wise addition of two large vectors `A` and `B` into a third vector `C`.
- To handle vectors that could potentially be larger than the maximum number of threads that can be launched in a single kernel launch, the kernel uses a *grid‑stride loop*.  This pattern iterates over the vector with a stride equal to the total number of threads launched (`blockDim.x * gridDim.x`), so that any size `N` that fits in device memory can be processed without needing multiple kernel launches.
- Boundary checks are performed inside the loop: the loop condition `i < N` ensures that a thread never reads or writes past the end of the array.
- The code uses 64‑bit unsigned integers (`size_t` and `unsigned long long`) for indices so that it can address vectors larger than 2^31-1 elements if the device has enough memory.
- Error checking macros (`CUDA_CHECK`) are provided to catch CUDA runtime errors.
- The program allocates memory with `cudaMalloc` and copies data from host to device.  For simplicity and to avoid separate copy operations, the data could also be allocated with unified memory (`cudaMallocManaged`), but `cudaMalloc` keeps the example focused on explicit memory handling.
- The main function takes an optional command line argument to specify the vector length; otherwise it defaults to 100 million elements (≈800 MB per vector).
- The kernel launch parameters are chosen to give a reasonable number of threads per block (e.g., 256) and a grid size that covers the entire vector, but due to the grid‑stride loop the exact grid size is not critical.
- Timing is performed with CUDA events to measure the kernel execution time.
- Finally, the program verifies the result on the host for correctness and frees all allocated resources.

This program is fully self‑contained and can be compiled with `nvcc`:
    nvcc -O2 -arch=sm_50 large_vector_add.cu -o large_vector_add
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Macro for checking CUDA errors following a CUDA API call
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",  \
                    __FILE__, __LINE__, cudaGetErrorString(err));          \
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

// Kernel performing element-wise addition with grid-stride loop and boundary checks
__global__ void vecAdd(const double *A, const double *B, double *C, size_t N)
{
    // Compute global thread index
    unsigned long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long stride = blockDim.x * gridDim.x;

    // Grid-stride loop to cover all elements
    for (unsigned long long i = idx; i < N; i += stride) {
        C[i] = A[i] + B[i];
    }
}

int main(int argc, char *argv[])
{
    // Default vector size: 100 million elements (~800 MB per vector)
    size_t N = 100000000ULL;
    if (argc > 1) {
        // Allow user to specify vector size via command line
        N = strtoull(argv[1], NULL, 10);
        if (N == 0) {
            fprintf(stderr, "Invalid vector size provided.\n");
            return EXIT_FAILURE;
        }
    }
    printf("Vector size: %llu elements (each %lu bytes)\n", N, sizeof(double));

    // Calculate total memory required (in bytes)
    size_t bytes = N * sizeof(double);
    printf("Total memory per vector: %zu MB\n", bytes / (1024 * 1024));

    // Allocate device memory for A, B, and C
    double *d_A = NULL;
    double *d_B = NULL;
    double *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C, bytes));

    // Allocate host memory for verification
    double *h_A = (double *)malloc(bytes);
    double *h_B = (double *)malloc(bytes);
    double *h_C = (double *)malloc(bytes);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host vectors with some values
    for (size_t i = 0; i < N; ++i) {
        h_A[i] = (double)i * 0.001;          // Example: 0.0, 0.001, 0.002, ...
        h_B[i] = (double)(N - i) * 0.002;    // Example: decreasing values
    }

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Define kernel launch parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    // Limit blocksPerGrid to a reasonable value to avoid exceeding grid dimension limits
    if (blocksPerGrid > 65535) {
        blocksPerGrid = 65535;
    }

    printf("Launching kernel with %d blocks of %d threads each.\n", blocksPerGrid, threadsPerBlock);

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // Record start event
    CUDA_CHECK(cudaEventRecord(start, 0));

    // Launch kernel
    vecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Record stop event
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    // Calculate elapsed time
    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Kernel execution time: %.3f ms\n", ms);

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // Verify a few elements
    bool success = true;
    for (size_t i = 0; i < 10 && i < N; ++i) {
        double expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-9) {
            fprintf(stderr, "Verification failed at index %zu: got %f, expected %f\n",
                    i, h_C[i], expected);
            success = false;
            break;
        }
    }
    if (success) {
        printf("Verification passed for first 10 elements.\n");
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return EXIT_SUCCESS;
}
```