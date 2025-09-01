/*
Aim of the program: Use `fprintf(stderr, ...)` to print the error message to standard error instead of standard out.

Thinking:
1. The goal is to demonstrate error handling in a CUDA C program by printing error messages to the standard error stream using fprintf(stderr, ...).
2. I will write a simple CUDA program that performs vector addition on the GPU. This program will allocate host and device memory, copy data to the device, launch a kernel, copy results back, and clean up.
3. After each CUDA runtime API call (cudaMalloc, cudaMemcpy, kernel launch, cudaDeviceSynchronize, cudaFree), I will check for errors using cudaGetLastError() or the returned cudaError_t. If an error is detected, I will print a descriptive message using fprintf(stderr, ...).
4. This approach ensures that error messages are routed to stderr, which is conventional for error reporting, making it easier to redirect stdout and stderr separately if needed.
5. I will include some normal informational messages printed to stdout via printf to show normal program flow.
6. The code is self-contained, uses only the CUDA runtime API, and compiles as a .cu file with nvcc.
*/

#include <stdio.h>
#include <cuda_runtime.h>

/* Simple CUDA kernel that adds two vectors */
__global__ void vectorAdd(const float *A, const float *B, float *C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N)
        C[i] = A[i] + B[i];
}

int main(void)
{
    const int N = 1 << 20;            // 1M elements
    size_t size = N * sizeof(float);
    float *h_A = NULL, *h_B = NULL, *h_C = NULL;
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    cudaError_t err;

    /* Allocate host memory */
    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size);
    h_C = (float *)malloc(size);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host vectors.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host arrays */
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    /* Allocate device memory */
    err = cudaMalloc((void **)&d_A, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector A (error code %d: %s)!\n", err, cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    err = cudaMalloc((void **)&d_B, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector B (error code %d: %s)!\n", err, cudaGetErrorString(err));
        cudaFree(d_A);
        return EXIT_FAILURE;
    }

    err = cudaMalloc((void **)&d_C, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device vector C (error code %d: %s)!\n", err, cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        return EXIT_FAILURE;
    }

    /* Copy host data to device */
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %d: %s)!\n", err, cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return EXIT_FAILURE;
    }

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %d: %s)!\n", err, cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return EXIT_FAILURE;
    }

    /* Launch kernel */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    /* Check for kernel launch errors */
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch vectorAdd kernel (error code %d: %s)!\n", err, cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return EXIT_FAILURE;
    }

    /* Wait for GPU to finish */
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA device synchronization failed (error code %d: %s)!\n", err, cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return EXIT_FAILURE;
    }

    /* Copy result back to host */
    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %d: %s)!\n", err, cudaGetErrorString(err));
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        return EXIT_FAILURE;
    }

    /* Simple verification */
    bool success = true;
    for (int i = 0; i < N; ++i) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            fprintf(stderr, "Verification failed at index %d: %f + %f != %f\n",
                    i, h_A[i], h_B[i], h_C[i]);
            success = false;
            break;
        }
    }

    if (success) {
        printf("Vector addition successful! All values verified.\n");
    } else {
        fprintf(stderr, "Vector addition failed verification.\n");
    }

    /* Clean up */
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
