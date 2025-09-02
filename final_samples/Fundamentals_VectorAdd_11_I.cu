/*
Use a grid-stride loop to initialize a very large array to a specific value.

Thought process:

1. The goal is to demonstrate the use of a grid‑stride loop in a CUDA kernel to
   efficiently initialize a large array on the device.  
2. We choose a 1‑dimensional float array of size N (e.g. 1 × 10⁸ elements) which
   is large enough to require multiple blocks and grid‑stride logic, but small
   enough to fit in typical GPU memory.  
3. The kernel will compute a global thread index (`idx`) and a stride equal to
   the total number of threads (`gridDim.x * blockDim.x`).  
   Each thread then walks through the array stepping by the stride until it
   reaches the end.  
4. Host code will allocate device memory with `cudaMalloc`, launch the kernel
   with a reasonable block size (256) and a grid size that covers all
   elements.  
5. After kernel execution we copy a few elements back to host memory with
   `cudaMemcpy` to verify the initialization.  
6. Basic error checking is added via a helper macro to simplify CUDA API
   error handling.  
7. The program prints a few sample values and a message indicating success.

Now the full CUDA C source file follows.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Macro for CUDA error checking */
#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        cudaError_t err = call;                                               \
        if (err != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(err));             \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

/* Kernel that uses a grid‑stride loop to initialize each element of the array to a specified value */
__global__ void initArrayKernel(float *arr, size_t N, float val)
{
    size_t idx   = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = gridDim.x * blockDim.x;

    for (size_t i = idx; i < N; i += stride) {
        arr[i] = val;
    }
}

int main(void)
{
    /* Size of the array (adjustable) */
    const size_t N = 100'000'000; /* 100 million elements (~381 MB for float) */

    /* Value to initialize the array with */
    const float initVal = 3.1415926535f;

    /* Allocate device memory */
    float *d_arr = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_arr, N * sizeof(float)));

    /* Determine block and grid dimensions */
    const int threadsPerBlock = 256;
    int blocksPerGrid = (int)((N + threadsPerBlock - 1) / threadsPerBlock);

    /* Launch kernel */
    initArrayKernel<<<blocksPerGrid, threadsPerBlock>>>(d_arr, N, initVal);

    /* Check for kernel launch errors */
    CHECK_CUDA(cudaGetLastError());

    /* Synchronize to ensure kernel completion */
    CHECK_CUDA(cudaDeviceSynchronize());

    /* Allocate host memory for verification */
    float *h_arr = (float *)malloc(N * sizeof(float));
    if (!h_arr) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        CHECK_CUDA(cudaFree(d_arr));
        return EXIT_FAILURE;
    }

    /* Copy a small portion back to host for verification */
    const size_t sampleSize = 10;
    CHECK_CUDA(cudaMemcpy(h_arr, d_arr, sampleSize * sizeof(float), cudaMemcpyDeviceToHost));

    /* Print sample values */
    printf("Sample of initialized values:\n");
    for (size_t i = 0; i < sampleSize; ++i) {
        printf("h_arr[%zu] = %f\n", i, h_arr[i]);
    }

    /* Clean up */
    free(h_arr);
    CHECK_CUDA(cudaFree(d_arr));

    printf("Array initialized successfully.\n");
    return EXIT_SUCCESS;
}
