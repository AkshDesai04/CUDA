```cuda
/* 
 * Aim of the program: "Is this a synchronous or asynchronous error?"
 *
 * Thinking:
 * The objective is to demonstrate the difference between synchronous and asynchronous
 * errors in CUDA programming. In CUDA, some errors are detected immediately
 * (synchronously) when the API call is made, while others (mostly related to
 * kernel execution) are detected only after a synchronization point or when the
 * error state is queried.  
 *
 * To illustrate this, the program will:
 * 1. Launch a simple kernel that intentionally writes out-of-bounds, which
 *    generates an asynchronous error (the error is only reported when the
 *    device is synchronized or when cudaGetLastError is called after the
 *    kernel launch).
 * 2. Show that calling cudaGetLastError right after the kernel launch does
 *    not report the out-of-bounds error because it hasn't executed yet.
 * 3. After a cudaDeviceSynchronize, the error will be reported.
 *
 * Additionally, the program will intentionally cause a synchronous error
 * by passing an invalid argument (e.g., passing a NULL pointer to a
 * kernel launch that expects a valid device pointer). The error will be
 * reported immediately by the API call.
 *
 * This demonstration helps clarify which errors are caught
 * synchronously and which require asynchronous handling.
 */

#include <stdio.h>
#include <cuda_runtime.h>

/* Kernel that writes out of bounds */
__global__ void out_of_bounds_kernel(int *d_arr, int size) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < size) {
        // Intentionally write beyond the allocated size
        d_arr[idx] = idx;
    } else {
        // Access an out-of-bounds index
        d_arr[size] = -1;  // This writes to d_arr[size], which is one element past the valid range
    }
}

int main() {
    const int N = 10;
    const int BLOCK_SIZE = 4;
    const int GRID_SIZE = 4;  // Total threads = 16 > N, ensuring out-of-bounds access

    int *d_arr = nullptr;
    cudaError_t err;

    /* Allocate device memory */
    err = cudaMalloc((void**)&d_arr, N * sizeof(int));
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    /* Launch kernel (asynchronous execution) */
    out_of_bounds_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_arr, N);

    /* Check for synchronous errors immediately after launch */
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        // This will NOT catch the out-of-bounds error because it's asynchronous
        fprintf(stderr, "Synchronous error detected after kernel launch: %s\n", cudaGetErrorString(err));
    } else {
        printf("No synchronous error after kernel launch.\n");
    }

    /* Now synchronize to force the GPU to finish execution */
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        // Here we catch the asynchronous out-of-bounds error
        fprintf(stderr, "Asynchronous error detected after cudaDeviceSynchronize: %s\n", cudaGetErrorString(err));
    } else {
        printf("No asynchronous error after cudaDeviceSynchronize.\n");
    }

    /* Demonstrate a synchronous error by passing a NULL pointer to a kernel */
    /* We create a dummy kernel that expects a valid pointer */
    __global__ void dummy_kernel(int *ptr) {
        int idx = threadIdx.x;
        ptr[idx] = idx;
    }

    /* Attempt to launch dummy_kernel with a NULL pointer */
    dummy_kernel<<<1, 1>>>(nullptr);  // This will generate a synchronous error

    /* Immediately check for errors */
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Synchronous error detected by passing NULL pointer: %s\n", cudaGetErrorString(err));
    } else {
        printf("No synchronous error when passing NULL pointer (unexpected).\n");
    }

    /* Clean up */
    cudaFree(d_arr);
    return 0;
}
```