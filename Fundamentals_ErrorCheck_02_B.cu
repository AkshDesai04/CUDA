/*
Aim of the program: Wrap all three `cudaMemcpy` calls (HtoD for A, HtoD for B, DtoH for C).
Thought process: 
1. We need a simple CUDA C program that demonstrates three cudaMemcpy operations:
   - Host to Device copy for array A
   - Host to Device copy for array B
   - Device to Host copy for array C
2. We'll create three host arrays, allocate device memory for them, perform the copies,
   launch a trivial kernel (vector addition) to generate C on the device, then copy C back.
3. We will add basic error checking for each CUDA call to ensure reliability.
4. Finally, we clean up all allocated memory and output a few results to verify correctness.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                         \
    do {                                                                         \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error in %s at line %d: %s\n",                 \
                    __FILE__, __LINE__, cudaGetErrorString(err));                \
            exit(EXIT_FAILURE);                                                  \
        }                                                                        \
    } while (0)

__global__ void vectorAdd(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void) {
    const int N = 1024;
    const size_t size = N * sizeof(float);

    /* Allocate host memory */
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    if (!h_A || !h_B || !h_C) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host arrays */
    for (int i = 0; i < N; ++i) {
        h_A[i] = (float)i;
        h_B[i] = (float)(N - i);
    }

    /* Allocate device memory */
    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc((void **)&d_A, size));
    CHECK_CUDA(cudaMalloc((void **)&d_B, size));
    CHECK_CUDA(cudaMalloc((void **)&d_C, size));

    /* Wrap all three cudaMemcpy calls */
    /* Host to Device copy for A */
    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));

    /* Host to Device copy for B */
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    /* Launch kernel to compute C = A + B on device */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Device to Host copy for C */
    CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    /* Print a few results to verify */
    printf("First 10 results of vector addition:\n");
    for (int i = 0; i < 10; ++i) {
        printf("%f + %f = %f\n", h_A[i], h_B[i], h_C[i]);
    }

    /* Clean up */
    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return EXIT_SUCCESS;
}
