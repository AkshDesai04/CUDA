/*
Copy only a portion of d_a to d_b using pointer arithmetic.
The goal of this program is to demonstrate how to copy a sub‑segment
of a device array to another device array by using pointer arithmetic
inside a CUDA kernel.  The program proceeds as follows:

1. Allocate two arrays on the host (`h_a` and `h_b`) of size `N`.
2. Fill `h_a` with consecutive integers for easy verification.
3. Allocate corresponding device arrays `d_a` and `d_b`.
4. Copy the entire `h_a` to `d_a`.
5. Define a kernel `copy_subarray` that takes a source pointer, a
   destination pointer, an offset into the source, and a length.
   Inside the kernel each thread copies a single element using
   pointer arithmetic:
   ```
   const int *src = d_a + offset + i;
   int *dst   = d_b + i;
   *dst = *src;
   ```
6. Launch the kernel with enough threads to cover the requested
   sub‑segment (`len` elements).
7. Copy the result back from `d_b` to `h_b`.
8. Print both the original array and the copied segment for verification.
9. Clean up memory on both host and device.

This program demonstrates how device memory can be manipulated
directly with pointers inside a CUDA kernel, and how to launch
a kernel that copies only a specific portion of an array.

The code below can be compiled with:
   nvcc copy_subarray.cu -o copy_subarray
and run with:
   ./copy_subarray
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Macro for checking CUDA errors following a CUDA API call
#define CHECK_CUDA(call)                                                        \
    do {                                                                         \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n",         \
                    __FILE__, __LINE__, err, cudaGetErrorName(err),             \
                    cudaGetErrorString(err));                                  \
            exit(EXIT_FAILURE);                                                 \
        }                                                                        \
    } while (0)

// Kernel that copies a subarray from src to dst using pointer arithmetic
__global__ void copy_subarray(const int *src, int *dst, int offset, int len)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        // Pointer arithmetic: compute the source and destination pointers
        const int *s = src + offset + i; // address of src[i + offset]
        int *d = dst + i;                // address of dst[i]
        *d = *s;                          // copy element
    }
}

int main(void)
{
    const int N = 100;        // Total size of the arrays
    const int offset = 20;    // Starting index for the copy
    const int len = 30;       // Number of elements to copy

    // Allocate host memory
    int *h_a = (int *)malloc(N * sizeof(int));
    int *h_b = (int *)malloc(N * sizeof(int));
    if (!h_a || !h_b) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize h_a with consecutive values and h_b with zeros
    for (int i = 0; i < N; ++i) {
        h_a[i] = i;
        h_b[i] = 0;
    }

    // Allocate device memory
    int *d_a = NULL;
    int *d_b = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_a, N * sizeof(int)));
    CHECK_CUDA(cudaMalloc((void **)&d_b, N * sizeof(int)));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_a, h_a, N * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_b, 0, N * sizeof(int))); // Ensure destination is zeroed

    // Launch kernel to copy a subarray from d_a to d_b
    int threadsPerBlock = 256;
    int blocksPerGrid = (len + threadsPerBlock - 1) / threadsPerBlock;
    copy_subarray<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, offset, len);
    CHECK_CUDA(cudaGetLastError()); // Check for kernel launch errors

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_b, d_b, N * sizeof(int), cudaMemcpyDeviceToHost));

    // Print original array
    printf("Original array h_a:\n");
    for (int i = 0; i < N; ++i) {
        printf("%4d", h_a[i]);
        if ((i + 1) % 10 == 0) printf("\n");
    }
    printf("\n");

    // Print copied segment
    printf("Copied segment into h_b (only elements 0 to len-1 are non-zero):\n");
    for (int i = 0; i < N; ++i) {
        printf("%4d", h_b[i]);
        if ((i + 1) % 10 == 0) printf("\n");
    }
    printf("\n");

    // Clean up
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    free(h_a);
    free(h_b);

    return EXIT_SUCCESS;
}
