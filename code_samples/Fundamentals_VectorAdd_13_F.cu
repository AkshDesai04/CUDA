```cuda
/* 
Aim of the program: Handle large vectors with boundary checks.

Thought process:
1. The task is to write a CUDA program that demonstrates how to safely operate on large vectors,
   ensuring that each GPU thread checks that it is within bounds before accessing array elements.
2. I chose a simple vector addition (C = A + B) because it illustrates the essential pattern
   of loading data from global memory, performing a computation, and storing the result,
   while also highlighting boundary checks.
3. Key considerations for large vectors:
   - The number of elements may not be a multiple of the block size. A simple bounds check
     (`if (idx < N)`) prevents out-of-bounds memory accesses for the remaining threads.
   - Memory allocation on the device uses `cudaMalloc` and should be checked for errors.
   - The kernel launch configuration (grid and block dimensions) is calculated to cover
     all elements, with `gridDim.x` computed as `(N + blockSize - 1) / blockSize`.
   - For clarity, error-checking macros (`CUDA_CHECK`) are provided to simplify debugging.
   - The program accepts an optional command-line argument specifying the vector size,
     defaulting to a large value (e.g., 10 million) if none is given.
4. Implementation steps:
   - Parse the optional vector size argument.
   - Allocate host arrays `h_A`, `h_B`, and `h_C` and initialize `h_A` and `h_B`.
   - Allocate device arrays `d_A`, `d_B`, `d_C`.
   - Copy `h_A` and `h_B` to the device.
   - Launch the kernel with an appropriate grid/block configuration.
   - Copy `d_C` back to the host.
   - Verify a few sample results and clean up resources.
5. The code is fully self-contained, compilable with `nvcc`, and uses standard CUDA
   API functions. No external libraries are required.

*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Error checking macro */
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",               \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));   \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

/* Kernel performing element-wise vector addition with boundary checks */
__global__ void vectorAdd(const float *A, const float *B, float *C, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

/* Host helper to initialize arrays with deterministic data */
void initArray(float *arr, size_t N, float offset)
{
    for (size_t i = 0; i < N; ++i) {
        arr[i] = (float)i + offset;
    }
}

int main(int argc, char *argv[])
{
    /* Default vector size: 10 million elements */
    size_t N = 10'000'000;
    if (argc > 1) {
        N = strtoul(argv[1], NULL, 10);
        if (N == 0) {
            fprintf(stderr, "Invalid vector size specified.\n");
            return EXIT_FAILURE;
        }
    }
    printf("Vector size: %zu\n", N);

    /* Allocate host memory */
    float *h_A = (float *)malloc(N * sizeof(float));
    float *h_B = (float *)malloc(N * sizeof(float));
    float *h_C = (float *)malloc(N * sizeof(float));
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host arrays */
    initArray(h_A, N, 0.0f);   // A[i] = i
    initArray(h_B, N, 1.0f);   // B[i] = i + 1

    /* Allocate device memory */
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_B, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_C, N * sizeof(float)));

    /* Copy data from host to device */
    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, N * sizeof(float), cudaMemcpyHostToDevice));

    /* Kernel launch configuration */
    const int blockSize = 256;
    int gridSize = (int)((N + blockSize - 1) / blockSize);

    /* Launch kernel */
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());  // Check for launch errors

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    /* Simple verification: check first, middle, last elements */
    size_t mid = N / 2;
    printf("Sample results:\n");
    printf("h_C[0]   = %f (expected %f)\n", h_C[0], h_A[0] + h_B[0]);
    printf("h_C[%zu] = %f (expected %f)\n", mid, h_C[mid], h_A[mid] + h_B[mid]);
    printf("h_C[%zu] = %f (expected %f)\n", N - 1, h_C[N - 1], h_A[N - 1] + h_B[N - 1]);

    /* Clean up */
    free(h_A);
    free(h_B);
    free(h_C);
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    printf("Vector addition completed successfully.\n");
    return EXIT_SUCCESS;
}
```