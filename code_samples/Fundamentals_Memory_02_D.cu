/*
Copy only the first half of the array back from the device and verify it.

The goal of this program is to demonstrate how to perform a partial
memory copy from device to host in CUDA. We allocate a host array,
initialize it, copy the entire array to device memory, launch a
kernel that doubles each element, and then copy back only the first
half of the array. After copying back, we verify that the first half
of the host array has the expected doubled values. The second half
remains unchanged because it was not copied back from the device.
This shows that partial copies work as intended and that host data
outside the copied range remains untouched.

The program follows these steps:

1. Define the array size `N` (must be even) and the number of threads per block.
2. Allocate a host array `h_data` of `float` and fill it with known values.
3. Allocate a device array `d_data`.
4. Copy the entire host array to device memory.
5. Launch a simple kernel that multiplies each element by 2.
6. Copy back only the first `N/2` elements from device to host.
7. Verify that the first half of the host array contains the doubled values.
   If any mismatch is found, the program reports an error.
8. Clean up all allocated memory.

The program uses basic CUDA error checking macros to catch any
issues with memory allocation, copy operations, or kernel launches.

Compile with: `nvcc partial_copy.cu -o partial_copy`
Run with: `./partial_copy`
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024                // Size of the array (must be even)
#define THREADS_PER_BLOCK 256 // Number of threads per block

// Simple CUDA kernel to double each element
__global__ void doubleElements(float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] *= 2.0f;
    }
}

// Error checking macro
#define CUDA_CHECK(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
                    __FILE__, __LINE__, err, cudaGetErrorName(err), \
                    cudaGetErrorString(err));                       \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

int main(void) {
    float *h_data = NULL;
    float *d_data = NULL;
    size_t bytes = N * sizeof(float);
    size_t half_bytes = (N / 2) * sizeof(float);

    // Allocate host memory
    h_data = (float *)malloc(bytes);
    if (!h_data) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Initialize host array with values 0,1,2,...
    for (int i = 0; i < N; ++i) {
        h_data[i] = (float)i;
    }

    // Allocate device memory
    CUDA_CHECK(cudaMalloc((void **)&d_data, bytes));

    // Copy entire array from host to device
    CUDA_CHECK(cudaMemcpy(d_data, h_data, bytes, cudaMemcpyHostToDevice));

    // Launch kernel to double each element
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    doubleElements<<<blocks, THREADS_PER_BLOCK>>>(d_data, N);
    CUDA_CHECK(cudaGetLastError()); // Check for kernel launch errors

    // Copy back only the first half of the array
    CUDA_CHECK(cudaMemcpy(h_data, d_data, half_bytes, cudaMemcpyDeviceToHost));

    // Verify the first half
    int errors = 0;
    for (int i = 0; i < N / 2; ++i) {
        float expected = (float)i * 2.0f;
        if (h_data[i] != expected) {
            fprintf(stderr, "Verification failed at index %d: expected %f, got %f\n",
                    i, expected, h_data[i]);
            errors++;
            if (errors > 10) { // Limit output
                fprintf(stderr, "More errors found...\n");
                break;
            }
        }
    }

    if (errors == 0) {
        printf("Verification succeeded: first half of the array is correct.\n");
    } else {
        printf("Verification failed with %d errors.\n", errors);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_data));
    free(h_data);

    return (errors == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
