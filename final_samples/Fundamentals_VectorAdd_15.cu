```cuda
/*
 * Aim of the program:
 * Implement a 'fused' kernel that performs two operations at once. For example, C[i] = A[i] + B[i] and D[i] = A[i] - B[i]. This can improve performance by increasing arithmetic intensity.
 *
 * Thinking:
 * 1. The goal is to showcase a fused kernel that simultaneously computes both the sum and the difference of two arrays. By doing both computations in a single kernel launch, we reduce kernel launch overhead and potentially improve memory bandwidth usage, as each thread reads A[i] and B[i] only once.
 *
 * 2. We will create a CUDA C program that:
 *    - Allocates host arrays A, B, C, D.
 *    - Initializes A and B with sample data.
 *    - Allocates corresponding device arrays.
 *    - Copies data from host to device.
 *    - Launches a fused kernel that computes C[i] = A[i] + B[i] and D[i] = A[i] - B[i] for each element.
 *    - Copies results back to host.
 *    - Validates a few elements to ensure correctness.
 *    - Measures the execution time of the kernel using CUDA events.
 *
 * 3. Implementation details:
 *    - Use single precision floats for simplicity.
 *    - Use a helper macro for CUDA error checking.
 *    - Choose a block size of 256 threads and compute the grid size accordingly.
 *    - The fused kernel will perform a bounds check before accessing memory.
 *
 * 4. This program is self-contained and can be compiled with nvcc:
 *      nvcc -o fused_add_sub fused_add_sub.cu
 *   Then run:
 *      ./fused_add_sub
 *
 * 5. The code demonstrates the concept of fusion and shows how arithmetic intensity can be increased by reducing memory traffic per operation.
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in file '%s' in line %i : %s.\n",   \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

/* Fused kernel: compute C = A + B and D = A - B */
__global__ void fusedAddSub(const float *A, const float *B, float *C, float *D, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float a = A[idx];
        float b = B[idx];
        C[idx] = a + b;
        D[idx] = a - b;
    }
}

int main()
{
    const int N = 1 << 20;          // 1M elements
    const size_t bytes = N * sizeof(float);

    // Allocate host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_C = (float*)malloc(bytes);
    float *h_D = (float*)malloc(bytes);

    if (!h_A || !h_B || !h_C || !h_D) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i) * 0.5f;
        h_B[i] = static_cast<float>(N - i) * 0.25f;
    }

    // Allocate device memory
    float *d_A, *d_B, *d_C, *d_D;
    CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_D, bytes));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Kernel launch parameters
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Timing the kernel execution
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    // Launch fused kernel
    fusedAddSub<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, d_D, N);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Kernel execution time: %.3f ms\n", milliseconds);

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_D, d_D, bytes, cudaMemcpyDeviceToHost));

    // Verify a few results
    bool success = true;
    for (int i = 0; i < 10; ++i) {
        float expectedC = h_A[i] + h_B[i];
        float expectedD = h_A[i] - h_B[i];
        if (fabs(h_C[i] - expectedC) > 1e-5 || fabs(h_D[i] - expectedD) > 1e-5) {
            fprintf(stderr, "Mismatch at index %d: C=%f expected %f, D=%f expected %f\n",
                    i, h_C[i], expectedC, h_D[i], expectedD);
            success = false;
            break;
        }
    }
    if (success) {
        printf("Verification passed for first 10 elements.\n");
    }

    // Clean up
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_D));
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_D);

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
```