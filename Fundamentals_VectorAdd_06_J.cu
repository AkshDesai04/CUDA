/*
Aim: Pass the scalar `a` via constant memory instead of as a kernel argument.

Thinking:
- Constant memory in CUDA is a small, cached memory space that is fast for broadcast read-only data across threads.
- To use constant memory, we declare a variable with the __constant__ qualifier on the device.
- The host can copy data into this constant memory using cudaMemcpyToSymbol.
- In the kernel, we can read the constant directly, eliminating the need to pass it as a kernel argument.
- This example demonstrates a simple vector multiplication kernel that uses a scalar `a` stored in constant memory.
- We include error checking, memory allocation, initialization, and a simple verification step.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Constant memory declaration for scalar 'a' */
__constant__ float d_a;

/* Kernel that multiplies each element of an input vector by the constant scalar 'a' */
__global__ void vecMul(const float *in, float *out, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        out[idx] = in[idx] * d_a;   // use constant memory directly
}

/* Simple CUDA error checking macro */
#define CHECK_CUDA(call)                                       \
    do {                                                       \
        cudaError_t err = (call);                              \
        if (err != cudaSuccess) {                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n",       \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                \
        }                                                      \
    } while (0)

int main(void)
{
    const int N = 1 << 20;  // 1,048,576 elements
    size_t size = N * sizeof(float);

    /* Allocate host memory */
    float *h_in  = (float *)malloc(size);
    float *h_out = (float *)malloc(size);
    if (!h_in || !h_out) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    /* Initialize input data */
    for (int i = 0; i < N; ++i)
        h_in[i] = (float)i;

    /* Allocate device memory */
    float *d_in  = NULL;
    float *d_out = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_in,  size));
    CHECK_CUDA(cudaMalloc((void **)&d_out, size));

    /* Copy input data to device */
    CHECK_CUDA(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    /* Set scalar 'a' and copy it to constant memory */
    float h_a = 2.5f;
    CHECK_CUDA(cudaMemcpyToSymbol(d_a, &h_a, sizeof(h_a), 0, cudaMemcpyHostToDevice));

    /* Launch kernel */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vecMul<<<blocksPerGrid, threadsPerBlock>>>(d_in, d_out, N);
    CHECK_CUDA(cudaGetLastError());    // Check for launch errors

    /* Copy result back to host */
    CHECK_CUDA(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));

    /* Simple verification: print first 5 results */
    printf("First 5 results (should be input * 2.5):\n");
    for (int i = 0; i < 5; ++i)
        printf("h_in[%d] = %f, h_out[%d] = %f\n", i, h_in[i], i, h_out[i]);

    /* Clean up */
    CHECK_CUDA(cudaFree(d_in));
    CHECK_CUDA(cudaFree(d_out));
    free(h_in);
    free(h_out);

    return EXIT_SUCCESS;
}
