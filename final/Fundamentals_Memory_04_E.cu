/*
 * Aim of the program:
 * What is the limitation of `cudaMemset`? (It can only set every byte to the same value).
 * Explain why you can't use it to initialize an array of floats to `1.0f`.
 *
 * Thinking:
 * `cudaMemset` works like the standard C `memset` – it writes the same byte value
 * into every byte of a contiguous memory region.  When you allocate an array of
 * floats on the device, each float occupies 4 bytes.  If you call
 * `cudaMemset(d_array, 1, N * sizeof(float))`, the value `1` is treated as a
 * single byte and is repeated for all bytes.  In IEEE‑754 a float value of
 * `1.0f` has the bit pattern `0x3F800000`.  Writing the byte value `0x01` into
 * all four bytes yields the pattern `0x01010101`, which is not `1.0f` but a
 * very small number (≈2.37e‑38).  Therefore `cudaMemset` cannot be used to set
 * floating‑point arrays to arbitrary values like `1.0f`.  To initialize floats
 * to `1.0f` you need to launch a kernel that writes the desired float value
 * into each element, or use `cudaMemset` only for values that are representable
 * by a single byte (e.g., zero).
 *
 * This program demonstrates the limitation by:
 * 1. Allocating an array of floats on the device.
 * 2. Using `cudaMemset` to try to set it to 1.0f and printing the first few
 *    values on the host (they will be incorrect).
 * 3. Launching a simple kernel that sets each element to 1.0f and printing the
 *    results again to confirm the correct initialization.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void setToOne(float *arr, size_t n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        arr[idx] = 1.0f;
}

void checkCudaError(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

int main(void)
{
    const size_t N = 16;              // small array for demo
    const size_t bytes = N * sizeof(float);

    float *d_array = NULL;
    checkCudaError(cudaMalloc((void **)&d_array, bytes), "cudaMalloc");

    /* Attempt to set the array to 1.0f using cudaMemset (incorrect) */
    checkCudaError(cudaMemset(d_array, 1, bytes), "cudaMemset");

    /* Copy back to host to inspect values */
    float h_array[ N ];
    checkCudaError(cudaMemcpy(h_array, d_array, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");

    printf("After cudaMemset(d_array, 1, bytes):\n");
    for (size_t i = 0; i < N; ++i)
    {
        printf("h_array[%zu] = %.8f\n", i, h_array[i]);
    }
    printf("\n");

    /* Now correctly initialize to 1.0f using a kernel */
    const int threadsPerBlock = 256;
    const int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    setToOne<<<blocks, threadsPerBlock>>>(d_array, N);
    checkCudaError(cudaGetLastError(), "Kernel launch");

    /* Copy back again */
    checkCudaError(cudaMemcpy(h_array, d_array, bytes, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");

    printf("After kernel setToOne<<<>>>(d_array, N):\n");
    for (size_t i = 0; i < N; ++i)
    {
        printf("h_array[%zu] = %.8f\n", i, h_array[i]);
    }
    printf("\n");

    /* Clean up */
    cudaFree(d_array);
    return 0;
}
