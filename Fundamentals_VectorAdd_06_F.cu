/*
Aim: Handle large vectors with boundary checks.

Thinking:
The task is to demonstrate a CUDA program that operates on very large vectors (potentially hundreds of millions of elements) while ensuring that every memory access is safe. 
In CUDA, when the number of elements exceeds the total number of launched threads, we typically use a grid‑stride loop: each thread processes multiple elements spaced by the total grid size. This pattern naturally includes a boundary check (i < N) inside the loop so that threads do not read or write past the end of the array.

Key considerations:
1. **Memory Allocation** – Use cudaMalloc for device memory and allocate host arrays with malloc or new. For very large vectors (e.g., 1e8 floats ≈ 400 MB per array) we must ensure the host and device can handle the size. 
2. **Error Checking** – Wrap CUDA API calls with a macro to check return status and abort on failure.
3. **Kernel Design** – A simple element‑wise addition kernel that uses the grid‑stride pattern. Each thread checks the bounds before accessing the arrays.
4. **Launch Configuration** – Compute the grid and block dimensions from the problem size. A typical block size is 256 or 512 threads.
5. **Performance** – The grid‑stride loop reduces the launch overhead and works for any vector length. For demonstration, we also use cudaEvent_t for timing the kernel execution.
6. **Verification** – After the kernel completes, copy the result back to the host and verify a few elements or compute the sum to confirm correctness.

The resulting program will:
- Parse a vector size (defaulting to 1e8 if not specified).
- Allocate and initialize two large input vectors on the host.
- Copy them to device memory.
- Launch the kernel with bounds checking.
- Copy the result back and perform a simple verification.
- Print elapsed time and a success message.

The code below implements all of this in a single .cu file, ready to be compiled with nvcc.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Error checking macro */
#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

/* Kernel: element-wise addition with grid-stride loop and boundary check */
__global__ void vectorAdd(const float *A, const float *B, float *C, size_t N)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = blockDim.x * gridDim.x;
    for (size_t i = idx; i < N; i += stride)
    {
        C[i] = A[i] + B[i];
    }
}

/* Host function to initialize an array with a simple pattern */
void initArray(float *array, size_t N, float value)
{
    for (size_t i = 0; i < N; ++i)
    {
        array[i] = value;
    }
}

int main(int argc, char *argv[])
{
    /* Default vector size: 100 million elements */
    size_t N = 100'000'000;
    if (argc > 1)
    {
        N = strtoull(argv[1], NULL, 10);
        if (N == 0)
        {
            fprintf(stderr, "Invalid vector size.\n");
            return EXIT_FAILURE;
        }
    }

    printf("Vector size: %zu\n", N);
    size_t bytes = N * sizeof(float);

    /* Allocate host memory */
    float *h_A = (float *)malloc(bytes);
    float *h_B = (float *)malloc(bytes);
    float *h_C = (float *)malloc(bytes);
    if (!h_A || !h_B || !h_C)
    {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host arrays */
    initArray(h_A, N, 1.0f); /* All elements = 1.0 */
    initArray(h_B, N, 2.0f); /* All elements = 2.0 */

    /* Allocate device memory */
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void **)&d_A, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_B, bytes));
    CHECK_CUDA(cudaMalloc((void **)&d_C, bytes));

    /* Copy data from host to device */
    CHECK_CUDA(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    /* Determine kernel launch configuration */
    const int blockSize = 256;
    int gridSize = (int)((N + blockSize - 1) / blockSize);

    /* Timing events */
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    /* Launch kernel */
    CHECK_CUDA(cudaEventRecord(start));
    vectorAdd<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaEventRecord(stop));

    /* Wait for kernel to finish */
    CHECK_CUDA(cudaEventSynchronize(stop));

    /* Compute elapsed time */
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Kernel execution time: %.3f ms\n", milliseconds);

    /* Copy result back to host */
    CHECK_CUDA(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    /* Verify a few elements */
    bool success = true;
    for (size_t i = 0; i < 10; ++i)
    {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5f)
        {
            fprintf(stderr, "Verification failed at index %zu: got %f, expected %f\n",
                    i, h_C[i], expected);
            success = false;
            break;
        }
    }
    if (success)
        printf("Verification passed for first 10 elements.\n");

    /* Clean up */
    free(h_A);
    free(h_B);
    free(h_C);
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return EXIT_SUCCESS;
}
