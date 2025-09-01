```c
/*
Aim of the program (word for word):
Explain what 'fused multiply-add' (FMA) is and why an operation like SAXPY is well-suited for it.

Thinking:
The program demonstrates a classic SAXPY operation (Single-Precision A·X Plus Y)
on the GPU using CUDA C.  SAXPY is defined as:
    y[i] = a * x[i] + y[i]
for i = 0 … N-1.

We implement the kernel using the CUDA FMA intrinsic `fmaf` (or `__fmaf_rn`) which
performs a fused multiply–add in one step: it multiplies two single‑precision
numbers and adds the result to a third, all in a single rounding.  This
has two important benefits:
1. **Accuracy** – the intermediate product a*x[i] is not rounded separately
   before the addition; the rounding happens only once on the final result.
2. **Performance** – many GPUs have a dedicated FMA instruction that
   executes faster and/or uses less power than performing a separate multiply
   and add.

Because SAXPY only requires one multiply and one add per element, it maps
perfectly onto an FMA.  Each thread performs the FMA in one instruction,
maximising throughput and reducing error.

The code below allocates vectors on the host and device, initializes them,
launches the kernel, and verifies the result.  The kernel uses `fmaf` to
carry out the operation.  The program also prints the elapsed GPU time and
the first few output values for sanity checking.
*/

#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

#define N (1<<20)            // Number of elements (1M)
#define BLOCK_SIZE 256       // Threads per block

// CUDA kernel performing SAXPY using fused multiply-add
__global__ void saxpy_fma(float *d_y, const float *d_x, float a, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Using the FMA intrinsic: y[idx] = a * x[idx] + y[idx]
        d_y[idx] = fmaf(a, d_x[idx], d_y[idx]);
    }
}

// Utility macro for checking CUDA errors
#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s)\\n", \
                    __FILE__, __LINE__, err, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

int main()
{
    float *h_x = (float*)malloc(N * sizeof(float));
    float *h_y = (float*)malloc(N * sizeof(float));
    float *h_y_ref = (float*)malloc(N * sizeof(float));

    if (!h_x || !h_y || !h_y_ref) {
        fprintf(stderr, "Failed to allocate host memory\\n");
        return EXIT_FAILURE;
    }

    // Initialize host arrays
    float a = 2.5f;
    for (int i = 0; i < N; ++i) {
        h_x[i] = (float)i * 0.01f;
        h_y[i] = (float)i * 0.02f;
        h_y_ref[i] = a * h_x[i] + h_y[i];  // Reference result
    }

    // Device memory allocation
    float *d_x, *d_y;
    CUDA_CHECK(cudaMalloc((void**)&d_x, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void**)&d_y, N * sizeof(float)));

    // Copy data to device
    CUDA_CHECK(cudaMemcpy(d_x, h_x, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, h_y, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));

    saxpy_fma<<<numBlocks, BLOCK_SIZE>>>(d_y, d_x, a, N);

    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("SAXPY FMA kernel time: %f ms\\n", ms);

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_y, d_y, N * sizeof(float), cudaMemcpyDeviceToHost));

    // Verify a few elements
    int mismatches = 0;
    for (int i = 0; i < 10; ++i) {
        if (fabs(h_y[i] - h_y_ref[i]) > 1e-5f) {
            printf("Mismatch at i=%d: GPU=%f, CPU=%f\\n", i, h_y[i], h_y_ref[i]);
            mismatches++;
        } else {
            printf("i=%d: GPU=%f, CPU=%f (OK)\\n", i, h_y[i], h_y_ref[i]);
        }
    }
    if (mismatches == 0) {
        printf("All checked values match the reference result.\\n");
    } else {
        printf("Found %d mismatches.\\n", mismatches);
    }

    // Clean up
    free(h_x);
    free(h_y);
    free(h_y_ref);
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return EXIT_SUCCESS;
}
```