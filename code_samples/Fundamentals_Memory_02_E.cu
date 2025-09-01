/*
 * Aim: Copy a sub-section of the device array (e.g., elements 100 to 149) to the beginning of the host array. This requires pointer arithmetic on both host and device pointers.
 *
 * Thinking:
 * 1. Allocate a device array (d_arr) and a host array (h_arr) of the same size.
 * 2. Fill the device array with known values (e.g., 0, 1, 2, …) so we can verify the copy.
 * 3. Use cudaMemcpy with pointer arithmetic: source pointer = d_arr + 100, destination pointer = h_arr + 0.
 * 4. The number of elements to copy is 50 (indices 100 to 149 inclusive).
 * 5. After copying, print the first 10 elements of the host array to confirm the operation.
 * 6. Include simple error‑checking macro to simplify CUDA API error handling.
 * 7. Free all allocated memory and device resources before exiting.
 */

#include <stdio.h>
#include <cuda_runtime.h>

/* Simple error checking macro */
#define CUDA_CHECK(call) do {                                        \
    cudaError_t err = call;                                         \
    if (err != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",           \
                #call, __FILE__, __LINE__, cudaGetErrorString(err));\
        exit(EXIT_FAILURE);                                         \
    }                                                               \
} while (0)

/* Simple kernel to initialize device array with its index values */
__global__ void init_kernel(int *d_arr, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        d_arr[idx] = idx;
    }
}

int main(void) {
    const int N = 200;           // total size of arrays
    const int start = 100;       // starting index of sub-section
    const int count = 50;        // number of elements to copy (100 to 149)

    /* Host array */
    int *h_arr = (int *)malloc(N * sizeof(int));
    if (!h_arr) {
        fprintf(stderr, "Failed to allocate host memory\n");
        return EXIT_FAILURE;
    }

    /* Device array */
    int *d_arr;
    CUDA_CHECK(cudaMalloc((void **)&d_arr, N * sizeof(int)));

    /* Launch kernel to initialize device array */
    int threadsPerBlock = 256;
    int blocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    init_kernel<<<blocks, threadsPerBlock>>>(d_arr, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    /* Copy sub-section from device to host using pointer arithmetic */
    // Source pointer: d_arr + start
    // Destination pointer: h_arr (beginning)
    CUDA_CHECK(cudaMemcpy(h_arr,
                          d_arr + start,
                          count * sizeof(int),
                          cudaMemcpyDeviceToHost));

    /* Print first 10 elements of host array to verify */
    printf("First 10 elements of host array after copy:\n");
    for (int i = 0; i < 10; ++i) {
        printf("%d ", h_arr[i]);
    }
    printf("\n");

    /* Optional: Verify entire copied range matches expected values */
    int ok = 1;
    for (int i = 0; i < count; ++i) {
        if (h_arr[i] != start + i) {
            printf("Mismatch at host[%d]: expected %d, got %d\n",
                   i, start + i, h_arr[i]);
            ok = 0;
            break;
        }
    }
    if (ok) {
        printf("Copy verified successfully.\n");
    }

    /* Clean up */
    free(h_arr);
    CUDA_CHECK(cudaFree(d_arr));

    return 0;
}
