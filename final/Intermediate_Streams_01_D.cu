/*
The aim of the program is: The kernel could be a simple vector squaring or SAXPY operation. 

Thinking:
- We choose to implement the SAXPY operation: y = a*x + y, where a is a scalar.
- Host code will allocate vectors x and y, fill them with sample data, and allocate corresponding device memory.
- A CUDA kernel will perform the SAXPY operation in parallel.
- After kernel execution, results are copied back to host and verified for correctness.
- Basic error checking is added for CUDA API calls.
- The program uses the CUDA runtime API and is written in plain C syntax suitable for a .cu file.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Error checking macro */
#define CHECK_CUDA(call)                                                     \
    do {                                                                     \
        cudaError_t err = call;                                              \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d - %s\n",                    \
                    __FILE__, __LINE__, cudaGetErrorString(err));            \
            exit(EXIT_FAILURE);                                             \
        }                                                                    \
    } while (0)

/* SAXPY kernel: y = a * x + y */
__global__ void saxpy_kernel(int n, float a, const float *x, float *y) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        y[idx] = a * x[idx] + y[idx];
    }
}

/* Host function to validate results */
int validate(const float *x, const float *y, int n, float a) {
    for (int i = 0; i < n; ++i) {
        float expected = a * x[i] + y[i];
        if (fabsf(expected - y[i]) > 1e-5f) {
            printf("Mismatch at index %d: host %f, device %f\n", i, expected, y[i]);
            return 0;
        }
    }
    return 1;
}

int main(void) {
    /* Problem size */
    const int N = 1 << 20; /* 1M elements */
    const float a = 2.5f;

    /* Allocate host memory */
    float *h_x = (float*)malloc(N * sizeof(float));
    float *h_y = (float*)malloc(N * sizeof(float));
    if (!h_x || !h_y) {
        fprintf(stderr, "Failed to allocate host arrays.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host arrays with sample data */
    for (int i = 0; i < N; ++i) {
        h_x[i] = (float)i / N;
        h_y[i] = (float)(N - i) / N;
    }

    /* Allocate device memory */
    float *d_x, *d_y;
    CHECK_CUDA(cudaMalloc((void**)&d_x, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc((void**)&d_y, N * sizeof(float)));

    /* Copy data from host to device */
    CHECK_CUDA(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice));

    /* Kernel launch parameters */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    /* Launch SAXPY kernel */
    saxpy_kernel<<<blocksPerGrid, threadsPerBlock>>>(N, a, d_x, d_y);
    CHECK_CUDA(cudaGetLastError());

    /* Copy result back to host */
    CHECK_CUDA(cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost));

    /* Validate results */
    int result = validate(h_x, h_y, N, a);
    if (result) {
        printf("SAXPY computation successful.\n");
    } else {
        printf("SAXPY computation failed.\n");
    }

    /* Clean up */
    free(h_x);
    free(h_y);
    CHECK_CUDA(cudaFree(d_x));
    CHECK_CUDA(cudaFree(d_y));

    return 0;
}
