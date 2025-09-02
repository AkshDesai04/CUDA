/*
 * Aim: In the loop, `thread_id` is the global index `blockIdx.x * blockDim.x + threadIdx.x`. `grid_size` is the total number of threads launched `gridDim.x * blockDim.x`. Implement this.
 *
 * Thinking:
 * We will write a simple CUDA program that demonstrates the use of global thread index and total grid size in a loop to perform an operation on an array.
 * The kernel will compute thread_id and grid_size. Then it will iterate over array indices starting from thread_id and stepping by grid_size.
 * For demonstration, we will perform a simple element-wise addition: C[i] = A[i] + B[i].
 * Host code will allocate arrays A and B, initialize them, copy to device, launch kernel with chosen block and grid dimensions,
 * copy result back, and verify correctness.
 * We will also print some elements to confirm the operation.
 * The code will include error checking for CUDA API calls.
 * It will be self-contained, compile with nvcc, and run on a CUDA-capable GPU.
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>

/* Macro for checking CUDA errors following a CUDA API call */
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",        \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

/* Kernel: element-wise addition C = A + B using global thread indexing */
__global__ void vectorAdd(const float *A, const float *B, float *C, int N)
{
    /* Compute global thread id and total number of threads in the grid */
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int grid_size = gridDim.x * blockDim.x;

    /* Iterate over array elements in a strided manner */
    for (int i = thread_id; i < N; i += grid_size)
    {
        C[i] = A[i] + B[i];
    }
}

/* Helper function to initialize an array with random floats */
void initArray(float *arr, int N)
{
    for (int i = 0; i < N; ++i)
    {
        arr[i] = (float)(rand()) / RAND_MAX;
    }
}

int main(void)
{
    const int N = 1 << 20;          /* Number of elements (1M) */
    const size_t size = N * sizeof(float);

    /* Allocate host memory */
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);

    if (!h_A || !h_B || !h_C)
    {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Seed RNG and initialize host arrays */
    srand((unsigned)time(NULL));
    initArray(h_A, N);
    initArray(h_B, N);

    /* Allocate device memory */
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_A, size));
    CUDA_CHECK(cudaMalloc((void **)&d_B, size));
    CUDA_CHECK(cudaMalloc((void **)&d_C, size));

    /* Copy data from host to device */
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    /* Determine block and grid dimensions */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    /* Launch kernel */
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    /* Verify a few results */
    int errors = 0;
    for (int i = 0; i < 10; ++i)
    {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5f)
        {
            fprintf(stderr, "Mismatch at index %d: host=%.5f, device=%.5f\n",
                    i, expected, h_C[i]);
            errors++;
        }
    }

    if (errors == 0)
    {
        printf("All tested elements match expected results.\n");
    }
    else
    {
        printf("There were %d mismatches.\n", errors);
    }

    /* Clean up */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
