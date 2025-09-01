/*
Aim: Deliberately corrupt one byte of the data on the host before verification to ensure your verification code works.

Thinking:
- The program will allocate a simple integer array on the host, initialize it with known values (index values), copy it to the device, run a trivial kernel that increments each element, copy the result back to the host, and then deliberately corrupt one element of the returned data to test the verification logic.
- The kernel will be a straightforward element-wise addition: `d_out[idx] = d_in[idx] + 1`.
- After the kernel finishes and the data is copied back, we will corrupt a single element on the host side (e.g., the element at index 512).  We will modify it to an incorrect value by adding an offset so that the verification routine will detect the mismatch.
- The verification routine will compare each element of the host output array to the expected value (`h_in[i] + 1`).  On any mismatch it will print a detailed error message indicating the index, the expected value, and the actual value.
- CUDA error checking will be performed after each CUDA API call to catch any runtime errors.
- The program is self-contained, uses only standard CUDA headers, and can be compiled with `nvcc` into an executable.
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1024          // Size of the array
#define BLOCK_SIZE 256  // Block size for kernel launch

// CUDA error checking macro
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error in file '%s' in line %i: %s.\n",    \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Kernel: add 1 to each element
__global__ void addKernel(int *d_out, const int *d_in, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_out[idx] = d_in[idx] + 1;
    }
}

int main(void)
{
    int *h_in = NULL;
    int *h_out = NULL;
    int *d_in = NULL;
    int *d_out = NULL;

    size_t size = N * sizeof(int);

    // Allocate host memory
    h_in = (int *)malloc(size);
    h_out = (int *)malloc(size);
    if (h_in == NULL || h_out == NULL) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize host input array
    for (int i = 0; i < N; ++i) {
        h_in[i] = i;
    }

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&d_in, size));
    CUDA_CHECK(cudaMalloc((void **)&d_out, size));

    // Copy input data to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice));

    // Launch kernel
    int numBlocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    addKernel<<<numBlocks, BLOCK_SIZE>>>(d_out, d_in, N);
    CUDA_CHECK(cudaGetLastError());

    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost));

    // Deliberately corrupt one byte of the data on the host before verification
    // Here we corrupt element at index 512 (arbitrary choice)
    int corruptIndex = 512;
    if (corruptIndex < N) {
        // Change the value to something incorrect
        h_out[corruptIndex] = h_in[corruptIndex] + 999; // wrong by 998
        printf("Corrupted element at index %d: set to %d (expected %d + 1).\n",
               corruptIndex, h_out[corruptIndex], h_in[corruptIndex] + 1);
    }

    // Verification
    int errors = 0;
    for (int i = 0; i < N; ++i) {
        int expected = h_in[i] + 1;
        if (h_out[i] != expected) {
            fprintf(stderr, "Verification failed at index %d: expected %d, got %d\n",
                    i, expected, h_out[i]);
            errors++;
            // Optionally break after first error
            // break;
        }
    }

    if (errors == 0) {
        printf("Verification passed. All elements match expected values.\n");
    } else {
        printf("Verification failed with %d error(s).\n", errors);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    free(h_in);
    free(h_out);

    return (errors == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
