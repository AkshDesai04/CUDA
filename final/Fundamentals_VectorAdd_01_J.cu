/*
Add error checking to all CUDA calls and the kernel launch.

The aim of this program is to demonstrate how to systematically add
error checking to every CUDA API call as well as to a kernel launch.
This includes:
- Allocation of device memory (cudaMalloc)
- Copying data between host and device (cudaMemcpy)
- Launching a kernel
- Synchronization and final copy back to the host
- Freeing device memory (cudaFree)
- Resetting the device at the end

The code defines a simple vector addition kernel. Each CUDA API
call is wrapped with a macro that invokes a helper function
`gpuAssert`, which prints a descriptive error message and aborts
execution if the call fails. After the kernel launch we explicitly
check for launch errors using `cudaGetLastError()` and perform a
synchronization with `cudaDeviceSynchronize()` to catch any runtime
errors that occur during kernel execution.

This pattern can be applied to any CUDA program to make debugging
and reliability much easier. */

#include <stdio.h>
#include <stdlib.h>

#define N 256

/* Macro for error checking */
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s in %s at line %d\n",
                cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

/* Simple vector addition kernel */
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
        C[i] = A[i] + B[i];
}

int main(void)
{
    int i;
    size_t size = N * sizeof(float);
    float *h_A = NULL;
    float *h_B = NULL;
    float *h_C = NULL;
    float *d_A = NULL;
    float *d_B = NULL;
    float *d_C = NULL;

    /* Allocate host memory */
    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    /* Initialize host arrays */
    for (i = 0; i < N; ++i)
    {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    /* Allocate device memory */
    cudaCheckError(cudaMalloc((void **)&d_A, size));
    cudaCheckError(cudaMalloc((void **)&d_B, size));
    cudaCheckError(cudaMalloc((void **)&d_C, size));

    /* Copy host memory to device */
    cudaCheckError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    /* Launch kernel */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    /* Check for launch errors */
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Kernel launch error: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    /* Wait for GPU to finish */
    cudaCheckError(cudaDeviceSynchronize());

    /* Copy result back to host */
    cudaCheckError(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    /* Verify result */
    for (i = 0; i < N; ++i)
    {
        if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at index %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
    printf("Test PASSED\n");

    /* Free device memory */
    cudaCheckError(cudaFree(d_A));
    cudaCheckError(cudaFree(d_B));
    cudaCheckError(cudaFree(d_C));

    /* Free host memory */
    free(h_A);
    free(h_B);
    free(h_C);

    /* Reset device */
    cudaCheckError(cudaDeviceReset());

    return 0;
}
