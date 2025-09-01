/*
Implement an in-place version: `data[i] = data[i] * data[i]`.

Thinking:
To perform the in-place squaring of an array on the GPU, we write a simple CUDA kernel
that takes the device pointer and its size. Each thread will compute the index
`idx = blockIdx.x * blockDim.x + threadIdx.x` and if this index is less than `N`,
it will replace the element at that position with its square. This operation
does not require any temporary buffers, so it is truly in‑place.

In the host code we:
1. Allocate and initialize a host array.
2. Allocate a device array and copy the host data to it.
3. Launch the kernel with an appropriate grid and block configuration.
4. Copy the result back to the host.
5. Verify and print a few results.
6. Clean up all allocated memory.

We also use CUDA error checking macros and simple timing with `cudaEvent_t`
to demonstrate how to measure kernel execution time. The code is written in
C/C++ with CUDA extensions and can be compiled with `nvcc`. 
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Macro to check CUDA API calls */
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t err = call;                                            \
        if (err != cudaSuccess) {                                          \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",              \
                    __func__, __FILE__, __LINE__, cudaGetErrorString(err));\
            exit(EXIT_FAILURE);                                            \
        }                                                                  \
    } while (0)

/* Kernel that squares each element in-place */
__global__ void square_inplace(float *data, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float val = data[idx];
        data[idx] = val * val;
    }
}

int main(void)
{
    /* Problem size */
    const int N = 1 << 20;   /* 1M elements */
    const int bytes = N * sizeof(float);

    /* Allocate and initialize host data */
    float *h_data = (float*)malloc(bytes);
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }
    for (int i = 0; i < N; ++i) {
        h_data[i] = (float)i / N;   /* values in [0,1) */
    }

    /* Allocate device memory */
    float *d_data = NULL;
    CUDA_CHECK(cudaMalloc((void**)&d_data, bytes));

    /* Copy data from host to device */
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    /* Determine block and grid sizes */
    const int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    /* Create events for timing */
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    /* Record start event */
    CUDA_CHECK(cudaEventRecord(start, 0));

    /* Launch kernel */
    square_inplace<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);
    CUDA_CHECK(cudaGetLastError());  /* check kernel launch */

    /* Record stop event */
    CUDA_CHECK(cudaEventRecord(stop, 0));
    CUDA_CHECK(cudaEventSynchronize(stop));

    /* Compute elapsed time */
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    printf("Kernel execution time: %.3f ms\n", ms);

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_data, d_data, bytes, cudaMemcpyDeviceToHost));

    /* Verify a few results */
    int errors = 0;
    for (int i = 0; i < 10; ++i) {
        float expected = (i / (float)N);
        expected = expected * expected;
        if (fabs(h_data[i] - expected) > 1e-5f) {
            fprintf(stderr, "Mismatch at index %d: got %f, expected %f\n",
                    i, h_data[i], expected);
            errors++;
        }
    }
    if (errors == 0) {
        printf("Verification passed for first 10 elements.\n");
    }

    /* Clean up */
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_data));
    free(h_data);

    return EXIT_SUCCESS;
}
