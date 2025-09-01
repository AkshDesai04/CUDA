/*
Aim: Implement a kernel for the SAXPY (Single-precision A*X Plus Y) operation. The operation is `Y = a*X + Y`, where `a` is a scalar float, and `X` and `Y` are vectors of floats. The scalar `a` should be passed as an argument to the kernel.

Thinking:
1. The SAXPY operation is a vector addition with scaling, which is embarrassingly parallel. Each element of Y can be updated independently: `Y[i] = a * X[i] + Y[i]`.
2. A CUDA kernel will perform this operation. The kernel takes the scalar `a`, pointers to device arrays `X` and `Y`, and the vector length `N`. Inside the kernel, each thread calculates its global index and updates `Y[idx]` if `idx < N` to avoid out‑of‑bounds accesses.
3. The host program will:
   - Allocate host memory for X and Y, initialize them with sample values.
   - Allocate device memory, copy data from host to device.
   - Launch the kernel with a configuration that covers all N elements.
   - Copy the updated Y back to the host.
   - Print a few elements to verify correctness.
   - Clean up device and host memory.
4. For simplicity, error checking macros will be minimal, but a basic `cudaCheckError` helper will be used to catch CUDA API failures.
5. The program will compile with `nvcc` and run on any CUDA‑capable device. The vector size is set to a moderate value (e.g., 1<<20) for demonstration, but can be changed easily.

The resulting .cu file is self‑contained and demonstrates the core concept of the SAXPY kernel in CUDA.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                            \
    {                                                               \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",        \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    }

/* Kernel: SAXPY - Y = a * X + Y */
__global__ void saxpy_kernel(float a, const float *X, float *Y, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        Y[idx] = a * X[idx] + Y[idx];
    }
}

int main(void)
{
    const int N = 1 << 20;          // 1,048,576 elements
    const float a = 2.5f;           // Scalar multiplier

    /* Allocate host memory */
    float *h_X = (float *)malloc(N * sizeof(float));
    float *h_Y = (float *)malloc(N * sizeof(float));
    if (!h_X || !h_Y) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize host vectors */
    for (int i = 0; i < N; ++i) {
        h_X[i] = (float)i * 0.01f;        // arbitrary data
        h_Y[i] = (float)i * 0.02f;        // arbitrary data
    }

    /* Allocate device memory */
    float *d_X = NULL;
    float *d_Y = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_X, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc((void **)&d_Y, N * sizeof(float)));

    /* Copy data from host to device */
    CUDA_CHECK(cudaMemcpy(d_X, h_X, N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Y, h_Y, N * sizeof(float), cudaMemcpyHostToDevice));

    /* Launch kernel */
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    saxpy_kernel<<<blocksPerGrid, threadsPerBlock>>>(a, d_X, d_Y, N);
    CUDA_CHECK(cudaGetLastError());  // Check kernel launch

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_Y, d_Y, N * sizeof(float), cudaMemcpyDeviceToHost));

    /* Print first 10 results for verification */
    printf("First 10 results of Y after SAXPY (a = %.2f):\n", a);
    for (int i = 0; i < 10; ++i) {
        printf("Y[%d] = %f\n", i, h_Y[i]);
    }

    /* Clean up */
    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_Y));
    free(h_X);
    free(h_Y);

    return EXIT_SUCCESS;
}
