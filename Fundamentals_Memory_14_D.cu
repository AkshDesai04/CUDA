```cu
/*
 * The error reported by `cuda-memcheck` is an asynchronous error. Show that without `cuda-memcheck`, the program may run and exit without any visible error, potentially producing wrong results.
 *
 * Thinking:
 * - We create a kernel that writes beyond the bounds of a device array.
 * - The write is performed by launching more threads than the allocated size.
 * - Because the error is asynchronous, the kernel launch returns immediately and the host does not immediately detect the out-of-bounds access.
 * - If we do not call cudaDeviceSynchronize() or check cudaGetLastError() after the kernel, the program continues executing.
 * - The out-of-bounds write corrupts adjacent memory, leading to incorrect results when we copy data back.
 * - When run with cuda-memcheck, an out-of-bounds memory access is reported.
 * - Without cuda-memcheck, the program runs silently, prints seemingly valid output, but the data is actually corrupted.
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

/* Kernel that writes out of bounds: we intentionally launch more threads than the buffer size.
 * The buffer is allocated to hold 2*N ints, but we treat it as two separate arrays: a[0..N-1] and b[N..2N-1].
 * The kernel writes to d_buf[threadIdx.x] for all 0 <= threadIdx.x < 2*N.
 * For indices >= N, this writes into the area that should belong to b, corrupting it.
 */
__global__ void write_out_of_bounds(int *d_buf, int N)
{
    int idx = threadIdx.x;          // 0 .. 2*N-1
    // No bounds check on idx; this will overflow for idx >= N
    d_buf[idx] = idx;
}

int main(void)
{
    const int N = 10;
    const int totalSize = 2 * N * sizeof(int);

    // Allocate device buffer for two arrays a and b (each of size N)
    int *d_buf = nullptr;
    cudaError_t err = cudaMalloc((void**)&d_buf, totalSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed: %s\n", cudaGetErrorString(err));
        return EXIT_FAILURE;
    }

    // Initialize buffer to zero
    err = cudaMemset(d_buf, 0, totalSize);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemset failed: %s\n", cudaGetErrorString(err));
        cudaFree(d_buf);
        return EXIT_FAILURE;
    }

    // Launch kernel with 2*N threads (10 threads will overflow the first array)
    write_out_of_bounds<<<1, 2 * N>>>(d_buf, N);

    // We do NOT call cudaDeviceSynchronize() or check cudaGetLastError() here,
    // so the asynchronous error may go undetected by the host.

    // Copy back data to host
    int *h_buf = new int[2 * N];
    err = cudaMemcpy(h_buf, d_buf, totalSize, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed: %s\n", cudaGetErrorString(err));
        delete[] h_buf;
        cudaFree(d_buf);
        return EXIT_FAILURE;
    }

    // Print results for array a (first N elements)
    printf("Array a (should be 0..%d):\n", N - 1);
    for (int i = 0; i < N; ++i) {
        printf("%d ", h_buf[i]);
    }
    printf("\n");

    // Print results for array b (next N elements)
    printf("Array b (should be untouched, but corrupted by overflow):\n");
    for (int i = N; i < 2 * N; ++i) {
        printf("%d ", h_buf[i]);
    }
    printf("\n");

    // Clean up
    delete[] h_buf;
    cudaFree(d_buf);

    // Optional: check for any lingering errors after kernel launch
    // This will catch the asynchronous error if we had called it
    // err = cudaGetLastError();
    // if (err != cudaSuccess) {
    //     fprintf(stderr, "Kernel error: %s\n", cudaGetErrorString(err));
    // }

    return EXIT_SUCCESS;
}
```