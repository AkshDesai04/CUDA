/*
 * Aim of the program: Modify the kernel to take a fourth argument, `int scalar`, and compute `b[i] = a[i] + scalar;`.
 *
 * Thinking:
 * The original requirement is to modify a CUDA kernel so that it accepts an additional
 * scalar integer value. The kernel should then add this scalar to each element of an
 * input array `a` and write the result to the output array `b`. The rest of the
 * program will demonstrate this by:
 *   1. Allocating and initializing a host array `a` with sequential integer values.
 *   2. Allocating corresponding device memory for `a` and `b`.
 *   3. Copying the host input to the device.
 *   4. Launching the kernel with a suitable grid/block configuration.
 *   5. Copying the result back to the host and printing a few elements for verification.
 *
 * Key decisions:
 * - Use `int` for all arrays to keep types consistent with the scalar argument.
 * - Provide a simple error checking macro `CUDA_CHECK` to catch CUDA API failures.
 * - Keep the kernel minimal: a straightforward bounds check and arithmetic.
 * - Choose a small `N` (e.g., 1000) and a small scalar value (e.g., 5) for demonstration.
 * - Print the first 10 results to verify correctness.
 *
 * The resulting code is a complete, compilable CUDA program in a single .cu file.
 */

#include <stdio.h>
#include <cuda_runtime.h>

/* Macro for checking CUDA errors */
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

/* Kernel that adds a scalar to each element of the input array */
__global__ void add_scalar(const int *a, int *b, int N, int scalar) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        b[idx] = a[idx] + scalar;
    }
}

int main(void) {
    const int N = 1000;          /* Number of elements */
    const int scalar = 5;        /* Scalar to add */

    /* Host allocations */
    int *h_a = (int *)malloc(N * sizeof(int));
    int *h_b = (int *)malloc(N * sizeof(int));
    if (h_a == NULL || h_b == NULL) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host input array */
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
    }

    /* Device allocations */
    int *d_a = NULL;
    int *d_b = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_a, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc((void **)&d_b, N * sizeof(int)));

    /* Copy input from host to device */
    CUDA_CHECK(cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice));

    /* Kernel launch parameters */
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    /* Launch the kernel */
    add_scalar<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, N, scalar);
    CUDA_CHECK(cudaGetLastError());  /* Check for kernel launch errors */

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_b, d_b, N * sizeof(int), cudaMemcpyDeviceToHost));

    /* Verify a few results */
    printf("First 10 results of a[i] + %d:\n", scalar);
    for (int i = 0; i < 10; ++i) {
        printf("b[%d] = %d\n", i, h_b[i]);
    }

    /* Clean up */
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    free(h_a);
    free(h_b);

    return EXIT_SUCCESS;
}
