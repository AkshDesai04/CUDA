/*
 Aim: Place all CUDA calls in your program on a separate line so the `CHECK` macro can wrap them cleanly.
 
 Thinking:
 The purpose of this exercise is to demonstrate a clean and maintainable way to handle CUDA runtime API errors by wrapping each call in a `CHECK` macro.  By placing each CUDA call on its own line and prepending the macro, we keep error handling consistent and avoid repetitive boilerplate.  The macro itself records the returned `cudaError_t`, checks if it indicates success, and if not, prints the error message and exits.  All host–device memory operations, kernel launches (followed by an explicit error check), and device synchronization calls are wrapped this way.  This pattern is especially useful in larger projects where many CUDA API calls are scattered throughout the code.
 
 In this example we perform a simple vector addition: we allocate three arrays on the device, copy input data from host to device, launch a kernel that adds two vectors element‑wise, copy the result back, and finally clean up.  Each CUDA API call (`cudaMalloc`, `cudaMemcpy`, `cudaFree`, `cudaGetLastError`, `cudaDeviceSynchronize`) is on a separate line and wrapped by `CHECK`.  The kernel launch itself is not a function call, so it is left on its own line followed by an error check.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Macro to check CUDA API calls */
#define CHECK(call)                                                   \
    do {                                                              \
        cudaError_t err = (call);                                     \
        if (err != cudaSuccess) {                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",              \
                    __FILE__, __LINE__, cudaGetErrorString(err));    \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

/* Kernel to add two vectors */
__global__ void vecAdd(const float *a, const float *b, float *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) c[idx] = a[idx] + b[idx];
}

int main(void) {
    const int N = 1 << 20; /* 1M elements */
    size_t bytes = N * sizeof(float);

    /* Host allocation */
    float *h_a = (float *)malloc(bytes);
    float *h_b = (float *)malloc(bytes);
    float *h_c = (float *)malloc(bytes);

    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host vectors\n");
        return EXIT_FAILURE;
    }

    /* Initialize host vectors */
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    /* Device allocation */
    float *d_a = NULL;
    CHECK(cudaMalloc((void **)&d_a, bytes));
    float *d_b = NULL;
    CHECK(cudaMalloc((void **)&d_b, bytes));
    float *d_c = NULL;
    CHECK(cudaMalloc((void **)&d_c, bytes));

    /* Copy data from host to device */
    CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    /* Kernel launch */
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    vecAdd<<<blocks, threads>>>(d_a, d_b, d_c, N);
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    /* Copy result back to host */
    CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    /* Verify result (optional) */
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        float expected = h_a[i] + h_b[i];
        if (fabs(h_c[i] - expected) > 1e-5) {
            if (errors < 10) {
                fprintf(stderr, "Mismatch at %d: %f != %f\n", i, h_c[i], expected);
            }
            errors++;
        }
    }
    if (errors == 0) {
        printf("Vector addition successful, no errors detected.\n");
    } else {
        printf("Vector addition found %d mismatches.\n", errors);
    }

    /* Free device memory */
    CHECK(cudaFree(d_a));
    CHECK(cudaFree(d_b));
    CHECK(cudaFree(d_c));

    /* Free host memory */
    free(h_a);
    free(h_b);
    free(h_c);

    return EXIT_SUCCESS;
}
