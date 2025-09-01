/*
After copying, use `cudaFree` to release the device memory.

The goal of this program is to illustrate a simple CUDA workflow:
1. Allocate host memory for two input arrays and an output array.
2. Allocate device memory for the same arrays.
3. Copy the input data from host to device (`cudaMemcpy`).
4. Launch a kernel that adds corresponding elements of the two input arrays.
5. Copy the result back from device to host.
6. Release the device memory with `cudaFree` (as requested).
7. Clean up host memory.

Key design decisions:
- Use a small vector addition example for clarity.
- Include basic error checking with a helper macro.
- Keep the kernel simple: element‑wise addition.
- All device allocations and frees are handled with `cudaMalloc` and `cudaFree`.
- Host memory is freed with `free` after use.

This demonstrates the lifecycle of device memory: allocation → copy → use → free, ensuring resources are correctly released.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Error checking macro */
#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",            \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                         \
        }                                                                \
    } while (0)

/* Simple kernel that adds two arrays element-wise */
__global__ void addKernel(const float *a, const float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main(void) {
    const int N = 1 << 20; // 1M elements
    const size_t size = N * sizeof(float);

    /* Allocate host memory */
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    if (!h_a || !h_b || !h_c) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host arrays */
    for (int i = 0; i < N; ++i) {
        h_a[i] = (float)i;
        h_b[i] = (float)(N - i);
    }

    /* Allocate device memory */
    float *d_a = NULL;
    float *d_b = NULL;
    float *d_c = NULL;
    CHECK_CUDA(cudaMalloc((void**)&d_a, size));
    CHECK_CUDA(cudaMalloc((void**)&d_b, size));
    CHECK_CUDA(cudaMalloc((void**)&d_c, size));

    /* Copy host input data to device */
    CHECK_CUDA(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));

    /* Launch kernel */
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    addKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CHECK_CUDA(cudaGetLastError());

    /* Copy result back to host */
    CHECK_CUDA(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    /* Verify a few results */
    bool success = true;
    for (int i = 0; i < 5; ++i) {
        float expected = h_a[i] + h_b[i];
        if (fabs(h_c[i] - expected) > 1e-5f) {
            fprintf(stderr, "Verification failed at index %d: got %f, expected %f\n",
                    i, h_c[i], expected);
            success = false;
            break;
        }
    }
    if (success) {
        printf("Vector addition succeeded. Sample output:\n");
        for (int i = 0; i < 5; ++i) {
            printf("%f + %f = %f\n", h_a[i], h_b[i], h_c[i]);
        }
    }

    /* Free device memory after copying back */
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));

    /* Free host memory */
    free(h_a);
    free(h_b);
    free(h_c);

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
