```cuda
/*
 * Aim of the program: Copy a sub-section of the device array (e.g., elements 100 to 149) to the beginning of the host array. This requires pointer arithmetic on both host and device pointers.
 *
 * Thinking:
 * 1. Allocate a device array of sufficient size (e.g., 200 ints).
 * 2. Fill the device array with known values using a simple kernel so we can verify the copy.
 * 3. Allocate a host array large enough to hold the 50-element sub-section.
 * 4. Use cudaMemcpy with pointer arithmetic: 
 *      - Source pointer on device: d_array + 100
 *      - Destination pointer on host: h_array (or h_array + 0)
 *    The number of bytes to copy is 50 * sizeof(int).
 * 5. Verify the result by printing the first 50 elements of the host array.
 * 6. Clean up memory.
 *
 * This demonstrates how to work with non‑zero offsets in both host and device pointers when using cudaMemcpy.
 */

#include <stdio.h>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,\
                    cudaGetErrorString(err));                              \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// Kernel to fill device array with its index values
__global__ void initArray(int *d_arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_arr[idx] = idx;
    }
}

int main(void) {
    const int N = 200;            // Size of device array
    const int SUBSTART = 100;     // Start index of sub-section
    const int SUBLEN = 50;        // Length of sub-section
    const int HOSTSIZE = SUBLEN;  // Host array size to hold the sub-section

    // Allocate host memory
    int *h_arr = (int *)malloc(HOSTSIZE * sizeof(int));
    if (h_arr == NULL) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    // Allocate device memory
    int *d_arr;
    CUDA_CHECK(cudaMalloc((void **)&d_arr, N * sizeof(int)));

    // Launch kernel to initialize device array
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    initArray<<<blocksPerGrid, threadsPerBlock>>>(d_arr, N);
    CUDA_CHECK(cudaGetLastError());  // Check for kernel launch errors
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for kernel to finish

    // Copy sub-section from device to host
    // Pointer arithmetic: source is d_arr + SUBSTART, destination is h_arr
    CUDA_CHECK(cudaMemcpy(h_arr,
                          d_arr + SUBSTART,              // device source pointer
                          SUBLEN * sizeof(int),          // number of bytes
                          cudaMemcpyDeviceToHost));

    // Print the copied elements
    printf("Copied elements from device[%d .. %d] to host[0 .. %d]:\n",
           SUBSTART, SUBSTART + SUBLEN - 1, SUBLEN - 1);
    for (int i = 0; i < SUBLEN; ++i) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    // Clean up
    CUDA_CHECK(cudaFree(d_arr));
    free(h_arr);

    // Reset device (optional)
    CUDA_CHECK(cudaDeviceReset());

    return 0;
}
```