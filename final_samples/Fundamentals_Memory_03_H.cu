/*
What happens if the source and destination pointers for a `cudaMemcpyDeviceToDevice` overlap? (Behavior is undefined, use `cudaMemmove` if needed).

The goal of this program is to illustrate the undefined behavior that can occur when performing a `cudaMemcpyDeviceToDevice` where the source and destination memory regions overlap.  CUDA’s documentation explicitly states that such an operation is undefined; the result can be corrupted data or other unexpected behavior.  To correctly handle overlapping copies we should use `cudaMemmove`.

In this example, we allocate a device array of 10 integers and initialise it from the host.  We then create two device pointers that point into this array such that the memory regions we copy overlap.  We first perform a standard `cudaMemcpyDeviceToDevice` on these overlapping ranges and copy the data back to the host to observe any corruption.  Next we perform the same copy using `cudaMemmove`, which is guaranteed to handle overlaps correctly, and copy the data back again to compare the results.

The output will show that the result of the `cudaMemcpyDeviceToDevice` is unreliable, whereas the `cudaMemmove` version yields the expected data.  This demonstrates the importance of choosing the right API for overlapping memory copies in CUDA.
*/

#include <stdio.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                         \
    do {                                                         \
        cudaError_t err = call;                                  \
        if (err != cudaSuccess) {                                \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",    \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                  \
        }                                                        \
    } while (0)

int main(void)
{
    const int N = 10;
    const int bytes = N * sizeof(int);

    // Host memory allocation
    int h_host[N];
    for (int i = 0; i < N; ++i)
        h_host[i] = i + 1;  // 1, 2, 3, ...

    // Device memory allocation
    int *d_dev;
    CHECK_CUDA(cudaMalloc((void **)&d_dev, bytes));

    // Copy data from host to device
    CHECK_CUDA(cudaMemcpy(d_dev, h_host, bytes, cudaMemcpyHostToDevice));

    // Create overlapping pointers
    // p_src points to elements 2,3,4,5 (indices 2-5)
    // p_dst points to elements 5,6,7,8 (indices 5-8)
    int *p_src = d_dev + 2;
    int *p_dst = d_dev + 5;

    // Number of ints to copy (4 ints = 16 bytes)
    size_t copy_bytes = 4 * sizeof(int);

    printf("=== Before copy ===\n");
    CHECK_CUDA(cudaMemcpy(h_host, d_dev, bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; ++i)
        printf("%2d ", h_host[i]);
    printf("\n\n");

    // Perform undefined overlapping copy
    printf("Performing cudaMemcpyDeviceToDevice with overlapping regions...\n");
    CHECK_CUDA(cudaMemcpy(p_dst, p_src, copy_bytes, cudaMemcpyDeviceToDevice));

    // Copy back to host to inspect result
    CHECK_CUDA(cudaMemcpy(h_host, d_dev, bytes, cudaMemcpyDeviceToHost));
    printf("=== After cudaMemcpyDeviceToDevice (overlap) ===\n");
    for (int i = 0; i < N; ++i)
        printf("%2d ", h_host[i]);
    printf("\n\n");

    // Reinitialize device array to original values
    CHECK_CUDA(cudaMemcpy(d_dev, h_host, bytes, cudaMemcpyHostToDevice));
    for (int i = 0; i < N; ++i)
        h_host[i] = i + 1;
    CHECK_CUDA(cudaMemcpy(d_dev, h_host, bytes, cudaMemcpyHostToDevice));

    // Perform correct overlapping copy using cudaMemmove
    printf("Performing cudaMemmove with overlapping regions...\n");
    CHECK_CUDA(cudaMemmove(p_dst, p_src, copy_bytes));

    // Copy back to host to inspect result
    CHECK_CUDA(cudaMemcpy(h_host, d_dev, bytes, cudaMemcpyDeviceToHost));
    printf("=== After cudaMemmove (overlap) ===\n");
    for (int i = 0; i < N; ++i)
        printf("%2d ", h_host[i]);
    printf("\n");

    // Clean up
    CHECK_CUDA(cudaFree(d_dev));

    return 0;
}
