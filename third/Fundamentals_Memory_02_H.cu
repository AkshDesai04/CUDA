/*
Use memcmp to verify the entire block of memory at once instead of a loop.
The idea of this program is to illustrate how a kernel can modify a block
of data on the device and then, on the host, the modified data can be
validated against an expected result using the standard C library
function memcmp.  Rather than iterating over each element to compare
expected and actual values, memcmp performs a single block comparison
which is both concise and efficient for large data sets.

The program performs the following steps:
1. Allocate and initialize a host array of integers with consecutive
   values.
2. Make a copy of the original data to preserve the initial state.
3. Allocate a device array and copy the original data to it.
4. Launch a simple kernel that increments each element by one.
5. Copy the modified data back to the host.
6. Build an expected result array on the host by incrementing the
   preserved original values.
7. Use memcmp to compare the entire block of the returned data with
   the expected data.
8. Report success or failure.

This demonstrates both basic CUDA memory operations and the use of
memcmp as a succinct way to validate large blocks of data.
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                          \
    do {                                                          \
        cudaError_t err = call;                                   \
        if (err != cudaSuccess) {                                 \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",     \
                    #call, __FILE__, __LINE__,                   \
                    cudaGetErrorString(err));                     \
            exit(EXIT_FAILURE);                                   \
        }                                                         \
    } while (0)

__global__ void inc_kernel(int *d_data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_data[idx] += 1;
    }
}

int main(void) {
    const int N = 1 << 20;          // 1M elements
    const size_t bytes = N * sizeof(int);

    // Allocate host memory
    int *h_orig = (int *)malloc(bytes);
    int *h_result = (int *)malloc(bytes);
    int *h_expected = (int *)malloc(bytes);
    if (!h_orig || !h_result || !h_expected) {
        fprintf(stderr, "Failed to allocate host memory.\n");
        return EXIT_FAILURE;
    }

    // Initialize original data
    for (int i = 0; i < N; ++i) {
        h_orig[i] = i;
    }

    // Preserve original copy for later verification
    memcpy(h_result, h_orig, bytes);

    // Allocate device memory
    int *d_data = NULL;
    CHECK_CUDA(cudaMalloc((void **)&d_data, bytes));

    // Copy original data to device
    CHECK_CUDA(cudaMemcpy(d_data, h_orig, bytes, cudaMemcpyHostToDevice));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    inc_kernel<<<blocks, threadsPerBlock>>>(d_data, N);
    CHECK_CUDA(cudaGetLastError());

    // Copy result back to host
    CHECK_CUDA(cudaMemcpy(h_result, d_data, bytes, cudaMemcpyDeviceToHost));

    // Build expected data on host
    for (int i = 0; i < N; ++i) {
        h_expected[i] = h_orig[i] + 1;
    }

    // Verify using memcmp
    int cmp = memcmp(h_result, h_expected, bytes);
    if (cmp == 0) {
        printf("Verification succeeded: all values match.\n");
    } else {
        printf("Verification failed: mismatch detected.\n");
    }

    // Clean up
    CHECK_CUDA(cudaFree(d_data));
    free(h_orig);
    free(h_result);
    free(h_expected);

    return EXIT_SUCCESS;
}
