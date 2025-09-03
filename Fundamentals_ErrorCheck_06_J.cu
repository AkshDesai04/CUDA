/*
In your helper function, if an error is found, print the error string using `cudaGetErrorString`.

This program demonstrates how to use a helper function to handle CUDA errors by printing the
error string provided by `cudaGetErrorString`. The program performs a simple vector addition
using a CUDA kernel. It allocates memory on both host and device, copies input data,
launches the kernel, copies the result back to the host, and then frees all allocated
memory. After every CUDA runtime API call and after the kernel launch, the program
calls `checkCudaError` to verify that the call succeeded. If any call fails, the helper
function prints a descriptive message along with the error string returned by
`cudaGetErrorString`, then aborts the program.

The helper function `checkCudaError` takes a `cudaError_t` returned by a CUDA call
and a string describing the operation that was attempted. If the error code is not
`cudaSuccess`, it prints the operation and the error string and exits with a non-zero
status. This pattern ensures that all CUDA errors are caught early and reported
in a clear, user-friendly way.

The code is written in CUDA C and compiled with `nvcc`. It includes a minimal
kernel `vectorAdd` that adds two input vectors element-wise. The main function
prepares example data, launches the kernel, and validates the result on the
host. The program is fully self-contained and can be compiled and executed on
any system with a CUDA-capable GPU and the CUDA Toolkit installed.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Helper function that checks CUDA errors.
   If an error is detected, it prints the provided message
   and the error string from cudaGetErrorString, then exits. */
void checkCudaError(cudaError_t err, const char* msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

/* Simple vector addition kernel: C = A + B */
__global__ void vectorAdd(const float* A, const float* B, float* C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

int main(void)
{
    const int N = 1 << 20;  // 1M elements
    size_t size = N * sizeof(float);

    /* Host allocations */
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host vectors */
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    /* Device allocations */
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    checkCudaError(cudaMalloc((void**)&d_A, size), "cudaMalloc d_A");
    checkCudaError(cudaMalloc((void**)&d_B, size), "cudaMalloc d_B");
    checkCudaError(cudaMalloc((void**)&d_C, size), "cudaMalloc d_C");

    /* Copy data from host to device */
    checkCudaError(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice), "cudaMemcpy A to d_A");
    checkCudaError(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice), "cudaMemcpy B to d_B");

    /* Launch kernel */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    checkCudaError(cudaGetLastError(), "Kernel launch error");
    checkCudaError(cudaDeviceSynchronize(), "Kernel execution error");

    /* Copy result back to host */
    checkCudaError(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost), "cudaMemcpy d_C to C");

    /* Verify results */
    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (fabs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5f) {
            success = false;
            fprintf(stderr, "Result verification failed at index %d: %f + %f != %f\n",
                    i, h_A[i], h_B[i], h_C[i]);
            break;
        }
    }

    if (success) {
        printf("Vector addition succeeded. Sample output:\n");
        for (int i = 0; i < 10; ++i)
            printf("%f + %f = %f\n", h_A[i], h_B[i], h_C[i]);
    } else {
        printf("Vector addition failed.\n");
    }

    /* Clean up */
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
