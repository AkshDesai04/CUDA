/*
Implement `C[i] = A[i] * scalar`.

In order to fulfill this request I will write a simple CUDA program that
multiplies each element of an input array A by a scalar value and stores
the result in array C.  The key parts of the implementation are:

1. A CUDA kernel `scalarMulKernel` that performs the element‑wise
   multiplication.  Each thread handles one array index, guarded by a
   bounds check to avoid out‑of‑range accesses.

2. Host code that:
   - Allocates host and device memory.
   - Initializes the input array with some test values.
   - Copies the data to the device.
   - Launches the kernel with an appropriate grid/block configuration.
   - Copies the result back to the host.
   - Verifies the result by printing a few elements.

3. A small utility macro `CUDA_CHECK` for error checking of CUDA API
   calls.

The program is self‑contained, uses only standard CUDA runtime API,
and can be compiled with `nvcc`.  No external dependencies are required.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Error checking macro */
#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",                \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));     \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

/* Kernel that multiplies each element of A by scalar and writes to C */
__global__ void scalarMulKernel(float *C, const float *A, float scalar, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        C[idx] = A[idx] * scalar;
    }
}

int main(void)
{
    const int N = 1024;                // Number of elements
    const float scalar = 3.1415f;      // Example scalar value

    /* Host memory allocation */
    float *h_A = (float *)malloc(N * sizeof(float));
    float *h_C = (float *)malloc(N * sizeof(float));
    if (!h_A || !h_C) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize input array A with some values (e.g., 0, 1, 2, ...) */
    for (int i = 0; i < N; ++i)
    {
        h_A[i] = (float)i;
    }

    /* Device memory allocation */
    float *d_A = NULL;
    float *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_A, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_C, N * sizeof(float)));

    /* Copy input data from host to device */
    CUDA_CHECK(cudaMemcpy(d_A, h_A, N * sizeof(float), cudaMemcpyHostToDevice));

    /* Kernel launch parameters */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    /* Launch the kernel */
    scalarMulKernel<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_A, scalar, N);
    CUDA_CHECK(cudaGetLastError());          // Check for launch errors
    CUDA_CHECK(cudaDeviceSynchronize());     // Ensure completion

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_C, d_C, N * sizeof(float), cudaMemcpyDeviceToHost));

    /* Verify a few results */
    printf("First 10 results of C[i] = A[i] * %f:\n", scalar);
    for (int i = 0; i < 10; ++i)
    {
        printf("C[%d] = %f\n", i, h_C[i]);
    }

    /* Clean up */
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_C);

    /* Reset device (optional) */
    CUDA_CHECK(cudaDeviceReset());

    return EXIT_SUCCESS;
}
