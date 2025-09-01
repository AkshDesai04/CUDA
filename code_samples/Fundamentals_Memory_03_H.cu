/*
What happens if the source and destination pointers for a `cudaMemcpyDeviceToDevice` overlap? (Behavior is undefined, use `cudaMemmove` if needed).

Thinking:
The goal of this program is to illustrate the undefined behavior that occurs when a cudaMemcpyDeviceToDevice operation is performed on overlapping memory regions.  We allocate a device array, initialize it from the host, then perform an overlapping copy using cudaMemcpyDeviceToDevice.  Because the source and destination overlap, the result is not guaranteed to be correct – the order in which the words are copied is unspecified and can lead to corrupted data.  For comparison, we then perform the same operation with cudaMemmove, which is designed to handle overlapping regions safely and produces the expected result.  By printing the array after each operation we can see the difference and understand why using cudaMemcpyDeviceToDevice for overlapping copies is unsafe.
*/

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                            \
    do {                                                            \
        cudaError_t err = call;                                     \
        if (err != cudaSuccess) {                                   \
            fprintf(stderr, "CUDA error in %s (%s:%d): %s\n",       \
                    #call, __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE);                                     \
        }                                                           \
    } while (0)

int main(void) {
    const int N = 10;
    size_t size = N * sizeof(int);

    int h[N];
    for (int i = 0; i < N; ++i) h[i] = i;  // 0,1,2,...,9

    int *d = nullptr;
    CHECK_CUDA(cudaMalloc(&d, size));

    // Copy host to device
    CHECK_CUDA(cudaMemcpy(d, h, size, cudaMemcpyHostToDevice));

    // Perform overlapping copy: copy 4 ints from d[2] to d[4]
    // This region overlaps (source indices 2-5, destination indices 4-7)
    CHECK_CUDA(cudaMemcpy(d + 4, d + 2, 4 * sizeof(int), cudaMemcpyDeviceToDevice));

    // Copy back to host to observe result
    int h_result1[N];
    CHECK_CUDA(cudaMemcpy(h_result1, d, size, cudaMemcpyDeviceToHost));

    printf("Result after cudaMemcpyDeviceToDevice with overlap:\n");
    for (int i = 0; i < N; ++i) {
        printf("%2d ", h_result1[i]);
    }
    printf("\n");

    // Reinitialize device from original host array
    CHECK_CUDA(cudaMemcpy(d, h, size, cudaMemcpyHostToDevice));

    // Perform overlapping copy using cudaMemmove (safe for overlap)
    CHECK_CUDA(cudaMemmove(d + 4, d + 2, 4 * sizeof(int)));

    // Copy back to host to observe result
    int h_result2[N];
    CHECK_CUDA(cudaMemcpy(h_result2, d, size, cudaMemcpyDeviceToHost));

    printf("Result after cudaMemmove with overlap:\n");
    for (int i = 0; i < N; ++i) {
        printf("%2d ", h_result2[i]);
    }
    printf("\n");

    CHECK_CUDA(cudaFree(d));

    return 0;
}
