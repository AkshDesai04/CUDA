```cuda
/*
The error reported by `cuda-memcheck` is an asynchronous error. Show that without `cuda-memcheck`, the program may run and exit without any visible error, potentially producing wrong results.

Thinking:
1. An asynchronous error in CUDA typically occurs during a kernel launch or an asynchronous memory operation.
2. The host will not immediately know that an error has happened until a synchronization or an explicit error-checking call is made.
3. To demonstrate this, we will:
   - Allocate a small device array (size 5).
   - Launch a kernel with 10 threads, intentionally writing out of bounds to trigger an asynchronous error.
   - After the kernel launch, we will copy the array back to the host using `cudaMemcpyAsync` on a stream.
   - We will *not* call `cudaDeviceSynchronize()` immediately; the program may exit without reporting the error.
4. For contrast, we include a call to `cudaGetLastError()` and `cudaDeviceSynchronize()` to show how the error can be detected.
5. Running this program under `cuda-memcheck` will immediately flag the out-of-bounds write, whereas a normal run may silently corrupt data and exit.
6. The output will print the host array; due to the error, some values may be incorrect or unchanged, illustrating the risk of silent failures.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK(call)                                                   \
    do {                                                              \
        cudaError_t err = call;                                       \
        if (err != cudaSuccess) {                                     \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",         \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                       \
        }                                                             \
    } while (0)

// Kernel that intentionally writes out of bounds.
__global__ void write_out_of_bounds(int *d_arr, int n)
{
    int idx = threadIdx.x;
    // Intentionally access beyond the array bounds.
    d_arr[idx] = idx * 10;   // For idx >= n, this is out-of-bounds.
}

int main()
{
    const int N = 5;          // Size of array on device.
    const int THREADS = 10;   // Number of threads (more than N).

    int *d_arr = NULL;
    int *h_arr = (int*)malloc(N * sizeof(int));

    // Allocate device memory.
    CHECK(cudaMalloc((void**)&d_arr, N * sizeof(int)));

    // Initialize device array to zeros.
    CHECK(cudaMemset(d_arr, 0, N * sizeof(int)));

    // Launch kernel with THREADS threads.
    write_out_of_bounds<<<1, THREADS>>>(d_arr, N);

    // Immediately query for errors. At this point, the kernel launch may not have completed,
    // but any launch failure will be caught here.
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "Error after kernel launch: %s\n", cudaGetErrorString(err));

    // Asynchronously copy data back to host on a stream.
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));
    CHECK(cudaMemcpyAsync(h_arr, d_arr, N * sizeof(int),
                          cudaMemcpyDeviceToHost, stream));

    // Without a synchronization point, the program may exit before the copy completes.
    // Uncomment the next line to see the error being detected.
    // CHECK(cudaDeviceSynchronize());

    // Check for errors after copy (this will block until copy completes).
    err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "Error after async copy: %s\n", cudaGetErrorString(err));

    // We purposely do NOT call cudaDeviceSynchronize() here to illustrate silent failure.
    // If we skip synchronization, the following printf may execute before the copy completes,
    // leading to incorrect or uninitialized output.

    // Wait for stream to finish to safely read data (for demonstration purposes).
    CHECK(cudaStreamSynchronize(stream));

    printf("Device array contents (host copy):\n");
    for (int i = 0; i < N; ++i) {
        printf("h_arr[%d] = %d\n", i, h_arr[i]);
    }

    // Cleanup
    CHECK(cudaFree(d_arr));
    free(h_arr);
    CHECK(cudaStreamDestroy(stream));

    return 0;
}
```