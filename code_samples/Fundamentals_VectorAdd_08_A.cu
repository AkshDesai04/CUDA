/*
Implement an in-place version: `data[i] += scalar_value`.

The goal of this CUDA program is to demonstrate how to perform an in-place update of an array on the GPU.  Each element of the array is incremented by a scalar value directly in device memory, so the result is stored back into the same array.  This avoids the need to allocate a separate output buffer and illustrates how to launch a kernel that writes to the same memory region it reads from.

Key considerations:
- Use a simple 1D grid and block configuration so that each thread processes one element.
- Guard against out-of-range indices because the total number of threads may exceed the array size.
- Perform necessary host–device memory copies before and after the kernel launch.
- Provide a small error‑checking helper macro to make debugging easier.
- Output the first few elements after the operation to verify correctness.

The program will:
1. Create a host array of floats initialized to sequential values.
2. Allocate a device array and copy the data over.
3. Launch a kernel that adds a scalar value (e.g., 5.0f) to each element in place.
4. Copy the updated array back to the host.
5. Print a few elements to confirm the operation succeeded.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

/* Simple macro for CUDA error checking */
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = (call);                                   \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s (%d): %s\n",          \
                    __FILE__, __LINE__, cudaGetErrorString(err));  \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

/* Kernel that adds a scalar value to each element in place */
__global__ void add_scalar(float *data, float scalar, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
    {
        data[idx] += scalar;
    }
}

int main(void)
{
    const int N = 1024;            /* Number of elements */
    const float scalar_value = 5.0f;

    /* Allocate host memory */
    float *h_data = (float *)malloc(N * sizeof(float));
    if (h_data == NULL)
    {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    /* Initialize host array with sequential values */
    for (int i = 0; i < N; ++i)
    {
        h_data[i] = (float)i;
    }

    /* Allocate device memory */
    float *d_data;
    CUDA_CHECK(cudaMalloc((void **)&d_data, N * sizeof(float)));

    /* Copy data from host to device */
    CUDA_CHECK(cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice));

    /* Determine execution configuration */
    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    /* Launch kernel */
    add_scalar<<<blocks_per_grid, threads_per_block>>>(d_data, scalar_value, N);
    CUDA_CHECK(cudaGetLastError());  /* Check for launch errors */

    /* Copy result back to host */
    CUDA_CHECK(cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost));

    /* Verify by printing first 10 elements */
    printf("First 10 elements after in-place addition of %f:\n", scalar_value);
    for (int i = 0; i < 10; ++i)
    {
        printf("h_data[%d] = %f\n", i, h_data[i]);
    }

    /* Clean up */
    CUDA_CHECK(cudaFree(d_data));
    free(h_data);

    return EXIT_SUCCESS;
}
